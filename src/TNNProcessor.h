#ifndef TNNProcessor_H
#define TNNProcessor_H

#include <string>
#include <iostream>
#include "tnn/core/tnn.h"
#include "tnn/core/macro.h"

namespace TNN_FOR_TRITION {

typedef enum{
  // run on cpu
  TNNComputeUnitsCPU = 0,
  // run on gpu, if failed run on cpu
  TNNComputeUnitsGPU = 1,
  // run on npu, if failed run on cpu
  TNNComputeUnitsNPU = 2,
} TNNComputeUnits;

class TNNProcessor{
public:
  TNNProcessor(const std::string &name,
                const int device_id,
                const std::vector<int> &nchw) : name_(name), device_id_(device_id), nchw_(nchw) {}
  //创建，backend创建一个服务实例，就会创建一个对应的TNNProcessor
  static bool Create(const std::string &name,
                      const int device_id,
                      const std::string &path,
                      const std::vector<int> &nchw,
                      std::shared_ptr<TNNProcessor> &processor);

  //根据input的值得到输出，并将*output指向输出。逻辑为转换指针为TNN_NS::Mat格式，前向计算，然后将结果转换为指针
  bool Run(const void *input, void **output);

  //下面两个函数在backend返回一个response的时调用，用于告知respon应该返回的张量形状和占用字节数
  //这里没有考虑多个output的情况，因为TNN一般只支持一个返回float类型tensor
  bool GetOutputShape(long **output_shape, int *output_dims_count); 
  bool GetOutputSize(int *output_byte_size) const; 

private:

  //TNN instance的Init，传入proto内容，模型内容，链接库目录，计算单元，在create时调用
  virtual TNN_NS::Status Init(const std::string &proto_content, const std::string &model_content,
                              const std::string &library_path, TNNComputeUnits units);
  //TNN instance的前向计算，在Run时调用
  virtual TNN_NS::Status Forward(const std::shared_ptr<TNN_NS::Mat> input, 
                                  std::shared_ptr<TNN_NS::Mat> &output);

private:
  // 来自triton的数据
  const std::string name_;
  const int device_id_;
  std::vector<int> nchw_;

  // TNN模型所需数据
  std::shared_ptr<TNN_NS::TNN> net_ = nullptr;
  std::shared_ptr<TNN_NS::Instance> instance_ = nullptr;
  TNN_NS::DeviceType device_type_ = TNN_NS::DEVICE_ARM;

  // 因为返回给triton_backend的是指针格式，哲学指针的内存管理存在processor内部，使用智能指针管理，以降低内存泄漏风险
  std::shared_ptr<TNN_NS::Mat> output_mat; 
  std::vector<long> output_shape_buffer; 

};

} // namespace TNN_FOR_TRITION
#endif //TNNProcessor_H