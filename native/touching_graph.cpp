#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <nanoflann.hpp>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <array>
namespace py = pybind11;
struct Cloud {
    const double* xyz; size_t n;
    size_t kdtree_get_point_count() const { return n; }
    double kdtree_get_pt(size_t i, size_t d) const { return xyz[3*i+d]; }
    template<class B> bool kdtree_get_bbox(B&) const { return false; }
};
struct Contacts {
    using DistanceType = double;
    const int64_t* labels;
    size_t source;
    double radius, bound;
    std::unordered_set<uint64_t>& edges;
    bool full() const { return true; }
    void sort() {}
    double worstDist() const { return bound; }
    bool addPoint(double distance, size_t target) {
        if (target > source && std::sqrt(distance) <= radius && labels[source] != labels[target]) {
            auto a = uint64_t(std::min(labels[source], labels[target]));
            auto b = uint64_t(std::max(labels[source], labels[target]));
            edges.insert((a << 32) | b);
        }
        return true;
    }
};
// Keep a compact labelled view of the spatial tree. Homogeneous branches can
// be skipped once their superpoint edge is known, including all own-label work.
struct LabelNode {
    std::array<double,3> low, high;
    int64_t label=-1;
    size_t left=0, right=0, begin=0, end=0;
};
template<class Tree> size_t annotate(const Tree& tree, typename Tree::NodePtr node,
        const Cloud& cloud, const int64_t* labels, std::vector<LabelNode>& nodes) {
    size_t index=nodes.size(); nodes.emplace_back();
    LabelNode item;
    item.low.fill(std::numeric_limits<double>::infinity());
    item.high.fill(-std::numeric_limits<double>::infinity());
    if (!node->child1) {
        item.begin=node->node_type.lr.left; item.end=node->node_type.lr.right;
        item.label=labels[tree.vAcc_[item.begin]];
        for (size_t k=item.begin;k<item.end;++k) {
            size_t p=tree.vAcc_[k];
            if (labels[p]!=item.label) item.label=-1;
            for (size_t d=0;d<3;++d) {
                item.low[d]=std::min(item.low[d],cloud.xyz[3*p+d]);
                item.high[d]=std::max(item.high[d],cloud.xyz[3*p+d]);
            }
        }
    } else {
        item.left=annotate(tree,node->child1,cloud,labels,nodes);
        item.right=annotate(tree,node->child2,cloud,labels,nodes);
        const auto& a=nodes[item.left]; const auto& b=nodes[item.right];
        item.label=a.label==b.label?a.label:-1;
        for (size_t d=0;d<3;++d) {
            item.low[d]=std::min(a.low[d],b.low[d]);
            item.high[d]=std::max(a.high[d],b.high[d]);
        }
    }
    nodes[index]=item; return index;
}
template<class Tree> void visit(const Tree& tree, const Cloud& cloud,
        const std::vector<LabelNode>& nodes, size_t index, Contacts& result) {
    const auto& node=nodes[index];
    auto source_label=result.labels[result.source];
    if (node.label>=0) {
        if (node.label==source_label) return;
        uint64_t key=(uint64_t(std::min(node.label,source_label))<<32)|uint64_t(std::max(node.label,source_label));
        if (result.edges.count(key)) return;
    }
    const double* query=cloud.xyz+3*result.source;
    double gap=0;
    for (size_t d=0;d<3;++d) {
        double delta=std::max(0.,std::max(node.low[d]-query[d],query[d]-node.high[d]));
        gap+=delta*delta;
    }
    if (gap>result.bound) return;
    if (node.end>node.begin) {
        for (size_t k=node.begin;k<node.end;++k) {
            size_t target=tree.vAcc_[k];
            if (target<=result.source || result.labels[target]==source_label) continue;
            double distance=0;
            for(size_t d=0;d<3;++d) { double delta=query[d]-cloud.xyz[3*target+d]; distance+=delta*delta; }
            result.addPoint(distance,target);
            if (node.label>=0 && std::sqrt(distance)<=result.radius) return;
        }
    } else {
        visit(tree,cloud,nodes,node.left,result);
        visit(tree,cloud,nodes,node.right,result);
    }
}
PYBIND11_MODULE(touching_graph, m) {
    m.def("edges", [](py::array_t<double, py::array::c_style | py::array::forcecast> xyz,
                      py::array_t<int64_t, py::array::c_style | py::array::forcecast> labels,
                      double radius, py::function progress) {
        if (xyz.ndim()!=2 || xyz.shape(1)!=3 || labels.ndim()!=1 || labels.size()!=xyz.shape(0)
            || !std::isfinite(radius) || radius<0) throw py::value_error("Invalid contact inputs");
        for (py::ssize_t i=0;i<labels.size();++i)
            if (labels.data()[i]<0 || uint64_t(labels.data()[i])>=(uint64_t(1)<<32))
                throw py::value_error("Labels must be compact uint32 IDs");
        for (py::ssize_t i=0;i<xyz.size();++i)
            if (!std::isfinite(xyz.data()[i])) throw py::value_error("Coordinates must be finite");
        Cloud cloud{xyz.data(), size_t(xyz.shape(0))};
        using Tree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double,Cloud>,Cloud,3,size_t>;
        std::unordered_set<uint64_t> unique;
        {
            py::gil_scoped_release release;
            Tree tree(3,cloud,nanoflann::KDTreeSingleIndexAdaptorParams(16));
            std::vector<LabelNode> nodes;
            if (cloud.n) annotate(tree,tree.root_node_,cloud,labels.data(),nodes);
            const double squared=radius*radius;
            const double bound=std::nextafter(squared + 32*std::numeric_limits<double>::epsilon()*squared,
                                              std::numeric_limits<double>::infinity());
            for (size_t start=0;start<cloud.n;start+=1024) {
                { py::gil_scoped_acquire acquire; progress("Streaming superpoint contacts",start,cloud.n); }
                for (size_t i=start;i<std::min(start+1024,cloud.n);++i) {
                    Contacts result{labels.data(),i,radius,bound,unique};
                    visit(tree,cloud,nodes,0,result);
                }
            }
        }
        std::vector<uint64_t> sorted(unique.begin(),unique.end());
        std::sort(sorted.begin(),sorted.end());
        py::array_t<uint64_t> output(sorted.size());
        std::copy(sorted.begin(),sorted.end(),output.mutable_data());
        return output;
    });
}
