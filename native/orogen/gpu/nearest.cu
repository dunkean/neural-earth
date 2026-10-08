// Exact k=4 nearest neighbours using a GPU uniform grid, with a distance bound.
// Expand cube shells until every unvisited bin is farther than the fourth hit.
extern "C" __global__ void nearest4(const float* points,const int* order,
 const int* offsets,const float* query,int* out,int n,int bins) {
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 float x=query[3*r],y=query[3*r+1],z=query[3*r+2];
 int cx=min(bins-1,max(0,(int)((x+1)*.5f*bins)));
 int cy=min(bins-1,max(0,(int)((y+1)*.5f*bins)));
 int cz=min(bins-1,max(0,(int)((z+1)*.5f*bins)));
 double distances[4]={1.e100,1.e100,1.e100,1.e100};int ids[4]={-1,-1,-1,-1};
 for(int radius=0;radius<bins;radius++) {
  int x0=max(0,cx-radius),x1=min(bins-1,cx+radius);
  int y0=max(0,cy-radius),y1=min(bins-1,cy+radius);
  int z0=max(0,cz-radius),z1=min(bins-1,cz+radius);
  for(int bz=z0;bz<=z1;bz++)for(int by=y0;by<=y1;by++)for(int bx=x0;bx<=x1;bx++){
   if(radius>0 && abs(bx-cx)<radius && abs(by-cy)<radius && abs(bz-cz)<radius)continue;
   int bin=bx+bins*(by+bins*bz);
   for(int i=offsets[bin];i<offsets[bin+1];i++){
    int id=order[i];double dx=(double)points[3*id]-x,dy=(double)points[3*id+1]-y,dz=(double)points[3*id+2]-z;
    double d=dx*dx+dy*dy+dz*dz;
    for(int k=0;k<4;k++)if(d<distances[k] || (d==distances[k]&&id<ids[k])){
     for(int j=3;j>k;j--){distances[j]=distances[j-1];ids[j]=ids[j-1];}
     distances[k]=d;ids[k]=id;break;
    }
   }
  }
  double lower=1.e100,step=2./bins;
  if(x0>0)lower=fmin(lower,x-(-1.+x0*step));
  if(x1<bins-1)lower=fmin(lower,-1.+(x1+1)*step-x);
  if(y0>0)lower=fmin(lower,y-(-1.+y0*step));
  if(y1<bins-1)lower=fmin(lower,-1.+(y1+1)*step-y);
  if(z0>0)lower=fmin(lower,z-(-1.+z0*step));
  if(z1<bins-1)lower=fmin(lower,-1.+(z1+1)*step-z);
  if(ids[3]>=0&&distances[3]<lower*lower)break;
 }
 for(int k=0;k<4;k++)out[4*r+k]=ids[k];
}
