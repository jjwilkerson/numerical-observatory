// Validation-only bridge; links COPIED reference library, never Python runtime.
#include "geodesic/metrics/numerical.hpp"
#include "geodesic/integrator.hpp"
#include "geodesic/initial_conditions.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <cmath>
using namespace geo;
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("usage: parity DATASET MODE probe|trace|raw|frame");
 std::string mode=argv[2],action=argv[3];
 if(action=="raw"){
  std::ifstream in(argv[1],std::ios::binary);std::size_t node;
  std::cout<<std::setprecision(17);while(std::cin>>node){in.seekg(node*80);double f[10];in.read(reinterpret_cast<char*>(f),80);if(!in)throw std::runtime_error("raw read failed");for(double x:f)std::cout<<x<<' ';std::cout<<'\n';}return 0;
 }
 auto spatial=mode=="linear"?SpatialInterpolation::linear:mode=="cubic"?SpatialInterpolation::cubic:SpatialInterpolation::cubic_amr_c1_hierarchy;
 auto m=std::make_unique<NumericalMetric>(argv[1],std::size_t(2)*1024*1024*1024,spatial);
 if(mode=="cubic_amr_c1_box"||mode=="cubic_amr_c1"){
  auto b=m->amr_hierarchy_transitions();if(b.size()!=1)throw std::runtime_error("box bridge expects two levels");
  if(mode=="cubic_amr_c1_box")m=std::make_unique<NumericalMetric>(argv[1],std::size_t(2)*1024*1024*1024,SpatialInterpolation::cubic_amr_c1_box,std::nullopt,b.front());
  else {auto box=b.front();PlanarAMRTransition plane{0,box.coarse_level,box.fine_level,box.inner_upper[0],box.outer_upper[0],true};m=std::make_unique<NumericalMetric>(argv[1],std::size_t(2)*1024*1024*1024,SpatialInterpolation::cubic_amr_c1,plane);}
 }
 std::cout<<std::setprecision(17);
 if(action=="probe"){
  Vec4 x;while(std::cin>>x[0]>>x[1]>>x[2]>>x[3]){
   try{auto g=m->evaluate(x);std::cout<<"OK ";for(auto row:g.inverse)for(double v:row)std::cout<<v<<' ';for(auto d:g.derivative)for(auto row:d)for(double v:row)std::cout<<v<<' ';
    if(mode!="linear"&&mode!="cubic")for(auto row:m->composite_support(x))if(row.required)std::cout<<"L"<<row.level<<' ';std::cout<<'\n';
   }catch(const std::exception&e){
    std::cout<<"ERROR "<<e.what()<<'\n';
   }
  }
 }else if(action=="eulerian"){
  Vec4 x;std::array<double,3> n;while(std::cin>>x[0]>>x[1]>>x[2]>>x[3]>>n[0]>>n[1]>>n[2]){
   auto f=eulerian_frame(*m,x);double length=std::hypot(n[0],n[1],n[2]);for(auto&v:n)v=-v/length;
   auto p=photon(*m,x,f,n);std::cout<<"OK ";for(double v:f.u)std::cout<<v<<' ';for(auto row:f.e)for(double v:row)std::cout<<v<<' ';for(int i=4;i<8;++i)std::cout<<-p[i]<<' ';std::cout<<'\n';
  }
 }else if(action=="frame"){
  Vec4 x;ObserverFrame f;while(std::cin>>x[0]>>x[1]>>x[2]>>x[3]){
   for(double&v:f.u)std::cin>>v;for(auto&row:f.e)for(double&v:row)std::cin>>v;
   auto g=invert(m->evaluate(x).inverse);validate_frame(g,f,1e-10);std::cout<<"OK ";for(auto row:g)for(double v:row)std::cout<<v<<' ';std::cout<<'\n';
  }
 }else if(action=="trace"){
  State state;IntegrationOptions opt;for(double&v:state)std::cin>>v;
  std::cin>>opt.max_lambda>>opt.max_step>>opt.rtol>>opt.atol;opt.initial_step=std::min(.01,opt.max_step);
  double radius=0,capture=0;std::cin>>radius>>capture;
  std::vector<EventSpec> events;if(radius>0)events.push_back(sphere("finite_background",{0,0,0},radius,CrossingDirection::increasing));
  if(capture>0)events.push_back(sphere("supplied_capture",{0,0,0},capture,CrossingDirection::decreasing));
  auto r=integrate(*m,state,opt,events);std::cout<<status_name(r.status)<<' '<<r.terminal_event<<'\n';
  for(auto sample:r.samples){std::cout<<sample.lambda<<' ';for(double v:sample.state)std::cout<<v<<' ';std::cout<<'\n';}
 }else throw std::runtime_error("unknown action");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
