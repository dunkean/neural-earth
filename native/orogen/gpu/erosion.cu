// Experimental Orogen graph erosion: resident deterministic gather passes.
// Retains stream-power, talus and latitude/elevation glacier controls. The
// numerical scheme replaces priority-heap carving and in-place deposition.
extern "C" __global__ void flood(const int* off,const int* adj,const unsigned char* sea,
 const float* original,const float* src,float* dst,int n,int* changed){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 if(sea[r]){dst[r]=original[r];return;}
 float low=1.e30f;for(int i=off[r];i<off[r+1];i++)low=fminf(low,fmaxf(0.f,src[adj[i]]));
 float next=fminf(src[r],fmaxf(original[r],low+1.e-7f));dst[r]=next;
 if(next<src[r])atomicExch(changed,1);
}
extern "C" __global__ void receivers(const int* off,const int* adj,const float* dist,
 const unsigned char* sea,const float* h,int* target,float* length,int n){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 target[r]=-1;length[r]=0;if(sea[r])return;
 float best=0;for(int i=off[r];i<off[r+1];i++){
  int nb=adj[i];float d=fmaxf(dist[i],1.e-6f),drop=(h[r]-fmaxf(h[nb],0.f))/d;
  if(drop>best){best=drop;target[r]=nb;length[r]=d;}
 }
}
extern "C" __global__ void accumulate(const int* off,const int* adj,const int* target,
 const unsigned char* sea,const float* src,float* dst,int n,int* changed){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 double area=sea[r]?0.:1.;for(int i=off[r];i<off[r+1];i++){int nb=adj[i];if(target[nb]==r)area+=src[nb];}
 dst[r]=(float)area;if(dst[r]!=src[r])atomicExch(changed,1);
}
extern "C" __global__ void incision(const int* target,const float* length,const unsigned char* sea,
 const float* original,const float* area,const float* src,float* dst,int n,float K,float m,float dt,int* changed){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 int nb=target[r];float next=original[r];
 if(!sea[r]&&nb>=0&&length[r]>0){float factor=K*powf(area[r],m)*dt/length[r];
  next=fmaxf(0.f,(original[r]+factor*fmaxf(src[nb],0.f))/(1.f+factor));}
 dst[r]=next;if(fabsf(next-src[r])>1.e-7f)atomicExch(changed,1);
}
extern "C" __global__ void deposition(const int* off,const int* adj,const int* target,const float* length,
 const unsigned char* sea,const float* original,const float* eroded,float* out,int n){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;if(sea[r]){out[r]=original[r];return;}
 int rec=target[r];float slope=rec>=0&&length[r]>0?fabsf(eroded[r]-fmaxf(eroded[rec],0.f))/length[r]:0.f;
 double deposit=0;for(int i=off[r];i<off[r+1];i++){int nb=adj[i];if(target[nb]==r)
  deposit+=fmaxf(0.f,original[nb]-eroded[nb])*.5f/(1.f+slope*50.f);}
 out[r]=fminf(original[r],eroded[r]+(float)deposit);
}
extern "C" __global__ void thermal(const int* off,const int* adj,const float* dist,const unsigned char* sea,
 const float* src,float* dst,int n,float talus,float rate){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;if(sea[r]){dst[r]=src[r];return;}
 double delta=0;for(int i=off[r];i<off[r+1];i++){int nb=adj[i];if(sea[nb])continue;
  float difference=src[r]-src[nb],excess=fmaxf(0.f,fabsf(difference)-talus*fmaxf(dist[i],1.e-6f));
  delta+=(difference>0?-1.:1.)*rate*.5*excess;}
 dst[r]=fmaxf(0.f,src[r]+(float)delta);
}
__device__ float ss(float x,float a,float b){float t=fmaxf(0.f,fminf(1.f,(x-a)/(b-a)));return t*t*(3.f-2.f*t);}
extern "C" __global__ void glaciers(const int* off,const int* adj,const float* dist,const unsigned char* sea,
 const float* xyz,const float* src,float* dst,int n,float strength,float scale){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;if(sea[r]){dst[r]=src[r];return;}
 float lat=fabsf(asinf(fmaxf(-1.f,fminf(1.f,xyz[3*r+1]))));
 float ice=fmaxf(ss(lat,1.5707963f-strength*3.14159265f/4.5f,1.5707963f),
  ss(src[r],.5f,.9f)*.3f*(.3f+.7f*ss(lat,3.14159265f/8.f,3.14159265f/3.f)))*strength;
 float drop=0;int coastal=0;double mean=0;int count=0;
 for(int i=off[r];i<off[r+1];i++){int nb=adj[i];if(sea[nb]){coastal=1;continue;}
  drop=fmaxf(drop,src[r]-src[nb]);mean+=src[nb];count++;}
 float carve=.02f*scale*ice*(.25f+fminf(1.f,drop*20.f));
 if(coastal)carve+=.015f*scale*ice;
 float valley=count?fmaxf(0.f,(float)(mean/count)-src[r]):0.f;
 dst[r]=fmaxf(0.f,src[r]-carve-valley*.1f*ice*scale);
}
