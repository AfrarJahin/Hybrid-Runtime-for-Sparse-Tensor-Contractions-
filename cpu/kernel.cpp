// cpu/kernel.cpp
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip> 
#include <map>
#include <string>
#include <vector>
#ifdef __APPLE__
#include <sys/stat.h>
#endif

using namespace std;

struct Timer {
  chrono::high_resolution_clock::time_point t0;
  void tic() { t0 = chrono::high_resolution_clock::now(); }
  double ms() const {
    auto t1 = chrono::high_resolution_clock::now();
    return chrono::duration<double, std::milli>(t1 - t0).count();
  }
};

static void load_binary(const string& path, vector<float>& buf) {
  ifstream f(path, ios::binary);
  if(!f) { cerr<<"Cannot open "<<path<<"\n"; exit(1); }
  f.seekg(0, ios::end);
  size_t bytes = size_t(f.tellg());
  f.seekg(0, ios::beg);
  buf.resize(bytes / sizeof(float));
  f.read(reinterpret_cast<char*>(buf.data()), bytes);
}

static string slurp(const string& path) {
  ifstream f(path);
  if(!f){ cerr<<"Cannot open "<<path<<"\n"; exit(1); }
  return string(istreambuf_iterator<char>(f), {});
}

// -------- Kernels: y = v^T W, v[N], W[N×OUT] row-major, y[OUT] --------

static inline void baseline(const float* __restrict v,
                            const float* __restrict W,
                            float* __restrict y, int N, int OUT) {
  for (int j = 0; j < OUT; ++j) {
    float acc = 0.f;
    #pragma omp simd
    for (int i = 0; i < N; ++i)
      acc += v[i] * W[i * OUT + j];
    y[j] = acc;
  }
}

template<int UF>
static inline void scalar_expand_unroll(const float* __restrict v,
                                        const float* __restrict W,
                                        float* __restrict y, int N, int OUT) {
  int j = 0;
  for (; j + UF <= OUT; j += UF) {
    float acc[UF]; for (int u=0; u<UF; ++u) acc[u]=0.f;
    #pragma omp simd
    for (int i = 0; i < N; ++i) {
      float x = v[i];
      #pragma unroll
      for (int u=0; u<UF; ++u) {
        acc[u] += x * W[i*OUT + (j+u)];
      }
    }
    for (int u=0; u<UF; ++u) y[j+u] = acc[u];
  }
  for (; j < OUT; ++j) {
    float acc = 0.f;
    #pragma omp simd
    for (int i = 0; i < N; ++i)
      acc += v[i] * W[i * OUT + j];
    y[j] = acc;
  }
}

static inline void loop_interchange(const float* __restrict v,
                                    const float* __restrict W,
                                    float* __restrict y, int N, int OUT) {
  for (int j = 0; j < OUT; ++j) y[j] = 0.f;
  for (int i = 0; i < N; ++i) {
    float x = v[i];
    const float* __restrict Wi = &W[i*OUT]; // contiguous row
    #pragma omp simd
    for (int j = 0; j < OUT; ++j)
      y[j] += x * Wi[j];
  }
}

static inline void tile_i(const float* __restrict v,
                          const float* __restrict W,
                          float* __restrict y, int N, int OUT, int Ti=2048) {
  for (int j = 0; j < OUT; ++j) y[j] = 0.f;
  for (int i0=0; i0<N; i0+=Ti) {
    int i1 = min(i0+Ti, N);
    for (int i=i0; i<i1; ++i) {
      float x = v[i];
      const float* __restrict Wi = &W[i*OUT];
      #pragma omp simd
      for (int j=0; j<OUT; ++j)
        y[j] += x * Wi[j];
    }
  }
}

enum Variant { BASE=0, SCALAR4, SCALAR8, INTERCH, TILEI };
static const char* vname(Variant v) {
  switch(v){
    case BASE: return "baseline";
    case SCALAR4: return "scalar_unroll_4";
    case SCALAR8: return "scalar_unroll_8";
    case INTERCH: return "loop_interchange";
    case TILEI: return "tile_i";
  }
  return "?";
}

static inline void run_one(const float* v, const float* W, float* y, int N, int OUT, Variant var){
  switch(var){
    case BASE:     baseline(v,W,y,N,OUT); break;
    case SCALAR4:  scalar_expand_unroll<4>(v,W,y,N,OUT); break;
    case SCALAR8:  scalar_expand_unroll<8>(v,W,y,N,OUT); break;
    case INTERCH:  loop_interchange(v,W,y,N,OUT); break;
    case TILEI:    tile_i(v,W,y,N,OUT,2048); break;
  }
}

static void ensure_dir(const string& p){
#ifdef __APPLE__
  struct stat st{};
  if (stat(p.c_str(), &st) != 0) mkdir(p.c_str(), 0755);
#endif
}

int main(){
  ensure_dir("../reports");

  // Parse sizes.json (simple)
  string js = slurp("../sizes.json");
  auto findi=[&](const string& key){
    auto p = js.find(key);
    auto c = js.find(':', p);
    auto e = js.find_first_of(",}", c+1);
    return stoi(js.substr(c+1, e-c-1));
  };
  const int N    = findi("\"N\"");
  const int OUT  = findi("\"OUT\"");
  const int CNT  = findi("\"count\"");

  vector<float> W; load_binary("../W.bin", W);
  vector<float> V; load_binary("../vectors.bin", V);
  if ((int)W.size() != N*OUT || (int)V.size() != CNT*N) {
    cerr<<"Size mismatch: W="<<W.size()<<" expected "<<N*OUT
        <<" V="<<V.size()<<" expected "<<CNT*N<<"\n";
    return 1;
  }
  vector<float> Y(CNT*OUT), tmp(OUT);

  // Warmup
  for (int n=0; n<min(8,CNT); ++n)
    run_one(&V[n*N], W.data(), tmp.data(), N, OUT, SCALAR8);

  const Variant variants[] = { BASE, SCALAR4, SCALAR8, INTERCH, TILEI };
  map<string,double> t_ms, gflops;

  for (auto var: variants) {
    Timer t; t.tic();
    #pragma omp parallel for schedule(dynamic)
    for (int n=0; n<CNT; ++n) {
      run_one(&V[n*N], W.data(), &Y[n*OUT], N, OUT, var);
    }
    double ms = t.ms();
    t_ms[vname(var)] = ms;
    double flops = double(CNT) * 2.0 * double(N) * double(OUT);
    gflops[vname(var)] = (flops/1e9) / (ms/1e3);
    cout<<setw(18)<<vname(var)<<"  time="<<ms<<" ms  GFLOP/s="<<gflops[vname(var)]<<"\n";
  }

  // pick best
  string best = "baseline";
  double best_ms = t_ms[best];
  for (auto &kv : t_ms) if (kv.second < best_ms){ best=kv.first; best_ms=kv.second; }
  cout<<"\nBEST VARIANT: "<<best<<"\n";

  // write JSON
  ofstream r("../reports/cpu_autotune.json");
  r<<"{\n"
   <<"  \"N\": "<<N<<", \"OUT\": "<<OUT<<", \"count\": "<<CNT<<",\n"
   <<"  \"results\": [\n";
  bool first=true;
  for (auto &kv : t_ms) {
    if(!first) r<<",\n"; first=false;
    r<<"    {\"name\":\""<<kv.first<<"\",\"ms\":"<<kv.second<<",\"gflops\":"<<gflops[kv.first]<<"}";
  }
  r<<"\n  ],\n  \"best\": \""<<best<<"\"\n}\n";
  r.close();
  return 0;
}
