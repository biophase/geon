// Small OCCT surface adapter. No CAD topology is persisted in geon documents.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Geom_Plane.hxx>
#include <Geom_Curve.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_RectangularTrimmedSurface.hxx>
#include <GeomAPI_IntSS.hxx>
#include <GeomAPI_IntCS.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <GeomAPI_ProjectPointOnCurve.hxx>
#include <TColgp_Array2OfPnt.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <Standard_Failure.hxx>
#include <gp_Ax3.hxx>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Vec = std::array<double, 3>;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
static Vec vec(const gp_Pnt& p) { return {p.X(), p.Y(), p.Z()}; }
static gp_Pnt point(const Vec& p) { return {p[0], p[1], p[2]}; }
static gp_Dir direction(const Vec& p) { return {p[0], p[1], p[2]}; }

struct Surface {
    Handle(Geom_Surface) geometry;
    std::array<double, 4> bounds;
    explicit Surface(const py::dict& spec) {
        const auto kind = py::cast<std::string>(spec["kind"]);
        bounds = py::cast<std::array<double, 4>>(spec["bounds"]); // u0,v0,u1,v1
        for (double x : bounds) if (!std::isfinite(x)) throw std::invalid_argument("Finite surface bounds required");
        if (bounds[2] <= bounds[0] || bounds[3] <= bounds[1]) throw std::invalid_argument("Empty surface domain");
        if (kind == "plane" || kind == "cylinder") {
            auto origin = py::cast<Vec>(spec["origin"]);
            gp_Ax3 axes(point(origin), direction(py::cast<Vec>(spec["normal"])), direction(py::cast<Vec>(spec["u"])));
            if (kind == "plane") geometry = new Geom_Plane(axes);
            else {
                double r = py::cast<double>(spec["radius"]);
                if (!std::isfinite(r) || r <= 0) throw std::invalid_argument("Positive cylinder radius required");
                geometry = new Geom_CylindricalSurface(axes, r);
            }
        } else if (kind == "bspline") {
            auto poles = py::cast<Array>(spec["poles"]);
            if (poles.ndim() != 3 || poles.shape(0) < 4 || poles.shape(1) < 4 || poles.shape(2) != 3 || poles.shape(0)>64 || poles.shape(1)>64)
                throw std::invalid_argument("Cubic poles must have shape (Nu,Nv,3), with 4 <= Nu,Nv <= 64");
            int nu=static_cast<int>(poles.shape(0)), nv=static_cast<int>(poles.shape(1));
            auto a = poles.unchecked<3>();
            TColgp_Array2OfPnt p(1, nu, 1, nv);
            for (int i=0;i<nu;++i) for(int j=0;j<nv;++j) {
                for(int k=0;k<3;++k) if(!std::isfinite(a(i,j,k))) throw std::invalid_argument("Nonfinite pole");
                p.SetValue(i+1,j+1,gp_Pnt(a(i,j,0),a(i,j,1),a(i,j,2)));
            }
            auto knots = [&](const char* name, int count) {
                std::vector<double> full = spec.contains(name) ? py::cast<std::vector<double>>(spec[name])
                    : std::vector<double>{0,0,0,0,1,1,1,1};
                if(full.size()!=static_cast<size_t>(count+4)) throw std::invalid_argument("Invalid cubic knot count");
                for(size_t i=0;i<full.size();++i) {
                    if(!std::isfinite(full[i]) || (i && full[i]<full[i-1]) || full[i]<0 || full[i]>1)
                        throw std::invalid_argument("Knots must be finite, sorted and normalized");
                }
                for(int i=0;i<4;++i) if(full[i]!=0 || full[full.size()-1-i]!=1)
                    throw std::invalid_argument("Cubic knot vector must be clamped at zero and one");
                std::vector<double> unique; std::vector<int> multiplicities;
                for(double value:full) {
                    if(unique.empty() || value!=unique.back()){unique.push_back(value);multiplicities.push_back(1);}
                    else ++multiplicities.back();
                }
                TColStd_Array1OfReal values(1,static_cast<int>(unique.size()));
                TColStd_Array1OfInteger mult(1,static_cast<int>(unique.size()));
                for(size_t i=0;i<unique.size();++i){values.SetValue(static_cast<int>(i)+1,unique[i]);mult.SetValue(static_cast<int>(i)+1,multiplicities[i]);}
                return std::make_pair(values,mult);
            };
            auto u=knots("u_knots",nu), v=knots("v_knots",nv);
            geometry = new Geom_BSplineSurface(p,u.first,v.first,u.second,v.second,3,3);
        } else throw std::invalid_argument("Unknown surface kind");
        geometry = new Geom_RectangularTrimmedSurface(geometry,bounds[0],bounds[2],bounds[1],bounds[3]);
    }
};

static std::pair<double,double> finite_range(const Handle(Geom_Curve)& curve, const Surface& a, const Surface& b) {
    double first=curve->FirstParameter(), last=curve->LastParameter();
    if (std::isfinite(first) && std::isfinite(last) && std::abs(first)<1e50 && std::abs(last)<1e50) return {first,last};
    first=std::numeric_limits<double>::infinity();last=-first;
    for (const Surface* s : {&a,&b}) for(int i=0;i<5;++i) for(int j=0;j<5;++j) {
        gp_Pnt p=s->geometry->Value(s->bounds[0]+(s->bounds[2]-s->bounds[0])*i/4.,s->bounds[1]+(s->bounds[3]-s->bounds[1])*j/4.);
        GeomAPI_ProjectPointOnCurve projection(p,curve);
        if(projection.NbPoints()) {
            double t=projection.LowerDistanceParameter();first=std::min(first,t);last=std::max(last,t);
        }
    }
    if(!std::isfinite(first)||!std::isfinite(last)||last<=first) throw std::runtime_error("Cannot bound intersection curve");
    double padding=(last-first)*.1;
    return {first-padding,last+padding};
}

static double chord_distance(const gp_Pnt& p,const gp_Pnt& a,const gp_Pnt& b) {
    gp_Vec ab(a,b),ap(a,p);
    double t=ab.SquareMagnitude()>0 ? std::clamp(ap.Dot(ab)/ab.SquareMagnitude(),0.,1.) : 0.;
    return p.Distance(a.Translated(ab*t));
}

static std::vector<Vec> sample_curve(const Handle(Geom_Curve)& curve,double first,double last,double step,double tolerance) {
    struct Span {double lo,hi;gp_Pnt a,b;int depth;};
    std::vector<Vec> result;
    // Initial subdivision also guards against a closed curve whose endpoints coincide.
    for(int part=0;part<16;++part) {
        double lo=first+(last-first)*part/16.,hi=first+(last-first)*(part+1)/16.;
        std::vector<Span> stack{{lo,hi,curve->Value(lo),curve->Value(hi),0}};
        while(!stack.empty()) {
            auto s=stack.back();stack.pop_back();
            double mid=(s.lo+s.hi)*.5;
            gp_Pnt m=curve->Value(mid),q1=curve->Value((s.lo+mid)*.5),q3=curve->Value((mid+s.hi)*.5);
            bool split=s.a.Distance(s.b)>step || std::max({chord_distance(m,s.a,s.b),chord_distance(q1,s.a,s.b),chord_distance(q3,s.a,s.b)})>tolerance;
            if(split) {
                if(s.depth>=24 || result.size()+stack.size()>1000000) throw std::runtime_error("Intersection sampling budget exceeded");
                stack.push_back({mid,s.hi,m,s.b,s.depth+1});stack.push_back({s.lo,mid,s.a,m,s.depth+1});
            } else {if(result.empty()) result.push_back(vec(s.a));result.push_back(vec(s.b));}
        }
    }
    return result;
}

static py::array_t<double> array_points(const std::vector<Vec>& points) {
    py::array_t<double> out({static_cast<py::ssize_t>(points.size()),py::ssize_t(3)});
    auto a=out.mutable_unchecked<2>();
    for(size_t i=0;i<points.size();++i)for(int k=0;k<3;++k)a(i,k)=points[i][k];
    return out;
}

PYBIND11_MODULE(surface_geometry,m) {
    m.doc()="Bounded OCCT 7.9.3 surfaces and intersections for reconstruction";
    m.attr("occt_version")="7.9.3";
    py::class_<Surface>(m,"Surface").def(py::init<const py::dict&>());
    m.def("intersections",[](const Surface& a,const Surface& b,double step,double tolerance,double chord) {
        if(!(step>0&&tolerance>0&&chord>0)) throw std::invalid_argument("Positive intersection tolerances required");
        std::vector<std::vector<Vec>> curves;
        {py::gil_scoped_release release;
         try {
            GeomAPI_IntSS intersection(a.geometry,b.geometry,tolerance);
            if(!intersection.IsDone()) throw std::runtime_error("OCCT surface intersection did not converge");
            for(int i=1;i<=intersection.NbLines();++i) {
                auto curve=intersection.Line(i);auto range=finite_range(curve,a,b);
                curves.push_back(sample_curve(curve,range.first,range.second,step,chord));
            }
         } catch(const Standard_Failure& e) {throw std::runtime_error(e.GetMessageString()?e.GetMessageString():"OCCT intersection failed");}}
        py::list out;for(auto& curve:curves)out.append(array_points(curve));return out;
    },py::arg("a"),py::arg("b"),py::arg("step"),py::arg("tolerance")=1e-7,py::arg("chord_tolerance")=.005);
    m.def("closest_points",[](const Surface& surface,Array points) {
        if(points.ndim()!=2||points.shape(1)!=3)throw std::invalid_argument("points must be (N,3)");
        auto p=points.unchecked<2>();std::vector<Vec> result(points.shape(0));
        {py::gil_scoped_release release;
         try {
            GeomAPI_ProjectPointOnSurf projection;
            auto b=surface.bounds;projection.Init(surface.geometry,b[0],b[2],b[1],b[3],1e-8);
            for(py::ssize_t i=0;i<points.shape(0);++i){
                projection.Perform(gp_Pnt(p(i,0),p(i,1),p(i,2)));
                if(!projection.IsDone()||!projection.NbPoints())throw std::runtime_error("OCCT surface projection failed");
                result[i]=vec(projection.NearestPoint());
            }
         }catch(const Standard_Failure& e){throw std::runtime_error(e.GetMessageString()?e.GetMessageString():"OCCT projection failed");}}
        return array_points(result);
    });
    m.def("triple_points",[](const Surface& a,const Surface& b,const Surface& c,double tolerance) {
        std::vector<Vec> result;
        {py::gil_scoped_release release;
         try {
            GeomAPI_IntSS intersection(a.geometry,b.geometry,tolerance);
            if(!intersection.IsDone())throw std::runtime_error("OCCT corner intersection failed");
            for(int i=1;i<=intersection.NbLines();++i){
                GeomAPI_IntCS corner(intersection.Line(i),c.geometry);
                if(!corner.IsDone())throw std::runtime_error("OCCT curve-surface corner intersection failed");
                for(int j=1;j<=corner.NbPoints();++j) result.push_back(vec(corner.Point(j)));
            }
         }catch(const Standard_Failure& e){throw std::runtime_error(e.GetMessageString()?e.GetMessageString():"OCCT corner intersection failed");}}
        return array_points(result);
    },py::arg("a"),py::arg("b"),py::arg("c"),py::arg("tolerance")=1e-7);
}
