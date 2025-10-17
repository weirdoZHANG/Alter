#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t x) {
    return static_cast<scalar_t>(1.0f / (1.0f + expf(-static_cast<float>(x))));
}

template <typename scalar_t>
__global__ void egu_single_layer_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ hx,
    const scalar_t* __restrict__ weight_ih,
    const scalar_t* __restrict__ weight_hh,
    const scalar_t* __restrict__ bias_ih,
    const scalar_t* __restrict__ bias_hh,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ hy,
    const int64_t B,
    const int64_t T,
    const int64_t Cin,
    const int64_t H,
    const bool batch_first,
    const bool is_backward_dir,
    const bool has_bias
) {
    const int64_t b = blockIdx.x;
    const int64_t h = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= B || h >= H) return;

    scalar_t h_prev = hx[b * H + h];

    for (int64_t t_seq = 0; t_seq < T; ++t_seq) {
        const int64_t t_original = is_backward_dir ? (T - 1 - t_seq) : t_seq;

        const int64_t x_base_idx = batch_first
            ? (b * T * Cin + t_original * Cin)
            : (t_original * B * Cin + b * Cin);
        const scalar_t* x_t = input + x_base_idx;

        scalar_t only_x = has_bias ? bias_ih[h] : static_cast<scalar_t>(0.0f);
        for (int64_t c = 0; c < Cin; ++c) {
            only_x += weight_ih[h * Cin + c] * x_t[c];
        }
        only_x *= sigmoid(only_x);

        scalar_t only_h = has_bias ? bias_hh[h] : static_cast<scalar_t>(0.0f);
        for (int64_t hp = 0; hp < H; ++hp) {
            only_h += weight_hh[h * H + hp] * h_prev;
        }
        only_h *= sigmoid(only_h);

        const scalar_t o_gate = sigmoid(only_x + only_h);
        h_prev = o_gate * only_x + (1.0f - o_gate) * only_h;

        const int64_t out_base_idx = batch_first
            ? (b * T * H + t_seq * H)
            : (t_seq * B * H + b * H);
        output[out_base_idx + h] = h_prev;
    }

    hy[b * H + h] = h_prev;
}

std::tuple<torch::Tensor, torch::Tensor> egu_forward_cuda(
    torch::Tensor input,
    torch::Tensor hx,
    std::vector<torch::Tensor> weight_ih_list,
    std::vector<torch::Tensor> weight_hh_list,
    std::vector<torch::Tensor> bias_ih_list = std::vector<torch::Tensor>(),
    std::vector<torch::Tensor> bias_hh_list = std::vector<torch::Tensor>(),
    int64_t num_layers = 1,
    double dropout_p = 0.0,
    bool training = true,
    bool bidirectional = false,
    bool batch_first = false
) {
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (got dim=", input.dim(), ")");
    TORCH_CHECK(hx.dim() == 3, "hx must be 3D ([layers*dir, B, H], got dim=", hx.dim(), ")");

    const int64_t num_dir = bidirectional ? 2 : 1;
    const int64_t expected_param_cnt = num_layers * num_dir;
    TORCH_CHECK(hx.size(0) == expected_param_cnt, "hx size[0] mismatch: expected ", expected_param_cnt, ", got ", hx.size(0));
    TORCH_CHECK(weight_ih_list.size() == expected_param_cnt, "weight_ih_list size mismatch: expected ", expected_param_cnt, ", got ", weight_ih_list.size());
    TORCH_CHECK(weight_hh_list.size() == expected_param_cnt, "weight_hh_list size mismatch: expected ", expected_param_cnt, ", got ", weight_hh_list.size());

    const bool has_bias = !bias_ih_list.empty() && !bias_hh_list.empty();
    TORCH_CHECK((bias_ih_list.empty() && bias_hh_list.empty()) || has_bias, "bias_ih_list and bias_hh_list must be both empty or non-empty");
    if (has_bias) {
        TORCH_CHECK(bias_ih_list.size() == expected_param_cnt && bias_hh_list.size() == expected_param_cnt, "Bias list size mismatch: expected ", expected_param_cnt);
    }

    const int64_t B = batch_first ? input.size(0) : input.size(1);
    const int64_t T = batch_first ? input.size(1) : input.size(0);
    const int64_t H = hx.size(2);
    auto tensor_opts = input.options().requires_grad(input.requires_grad());

    torch::Tensor output = torch::empty(
        batch_first ? std::vector<int64_t>{B, T, H * num_dir} : std::vector<int64_t>{T, B, H * num_dir},
        tensor_opts
    );
    torch::Tensor hy = torch::empty({expected_param_cnt, B, H}, tensor_opts);

    input = input.contiguous();
    hx = hx.contiguous();

    std::vector<torch::Tensor> wih_dev, whh_dev, bih_dev, bhh_dev;
    for (int64_t i = 0; i < expected_param_cnt; ++i) {
        wih_dev.push_back(weight_ih_list[i].contiguous().to(input.device()));
        whh_dev.push_back(weight_hh_list[i].contiguous().to(input.device()));
        if (has_bias) {
            bih_dev.push_back(bias_ih_list[i].contiguous().to(input.device()));
            bhh_dev.push_back(bias_hh_list[i].contiguous().to(input.device()));
        }
    }

    torch::Tensor curr_input = input;
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        const int64_t curr_Cin = curr_input.size(2);
        TORCH_CHECK(curr_Cin > 0, "Invalid curr_Cin (", curr_Cin, ") at layer ", layer);

        std::vector<torch::Tensor> dir_outputs;

        for (int64_t dir = 0; dir < num_dir; ++dir) {
            const int64_t param_idx = layer * num_dir + dir;
            const bool is_backward = (dir == 1) && bidirectional;

            auto wih = wih_dev[param_idx];
            auto whh = whh_dev[param_idx];
            torch::Tensor bih = has_bias ? bih_dev[param_idx] : torch::Tensor();
            torch::Tensor bhh = has_bias ? bhh_dev[param_idx] : torch::Tensor();
            auto hx_dir = hx[param_idx].contiguous().to(input.device());

            TORCH_CHECK(wih.size(0) == H && wih.size(1) == curr_Cin, "weight_ih[", param_idx, "] dim mismatch: expected [", H, ",", curr_Cin, "], got [", wih.size(0), ",", wih.size(1), "]");
            TORCH_CHECK(whh.size(0) == H && whh.size(1) == H, "weight_hh[", param_idx, "] dim mismatch: expected [", H, ",", H, "], got [", whh.size(0), ",", whh.size(1), "]");

            auto output_dir = output.slice(-1, dir * H, (dir + 1) * H).contiguous();

            const int threads_per_block = 256;
            const int blocks_h = (H + threads_per_block - 1) / threads_per_block;
            const dim3 grid(B, blocks_h);
            const dim3 block(threads_per_block);

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(curr_input.scalar_type(), "egu_forward", ([&] {
                egu_single_layer_forward_kernel<scalar_t><<<grid, block>>>(
                    curr_input.data_ptr<scalar_t>(),
                    hx_dir.data_ptr<scalar_t>(),
                    wih.data_ptr<scalar_t>(),
                    whh.data_ptr<scalar_t>(),
                    has_bias ? bih.data_ptr<scalar_t>() : nullptr,
                    has_bias ? bhh.data_ptr<scalar_t>() : nullptr,
                    output_dir.data_ptr<scalar_t>(),
                    hy[param_idx].data_ptr<scalar_t>(),
                    B, T, curr_Cin, H,
                    batch_first, is_backward, has_bias
                );
            }));

            dir_outputs.push_back(output_dir);
        }

        curr_input = (num_dir == 1) ? dir_outputs[0].contiguous() : torch::cat(dir_outputs, -1).contiguous();

        if (layer < num_layers - 1 && training && dropout_p > 0.0 && dropout_p < 1.0) {
            curr_input = torch::nn::functional::dropout(
                curr_input, torch::nn::functional::DropoutFuncOptions().p(dropout_p).training(training)
            );
        }
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return std::make_tuple(output, hy);
}

PYBIND11_MODULE(egu_impl_cuda, m) {
    m.doc() = "EGU CUDA Forward Propagation";
    m.def(
        "forward", &egu_forward_cuda,
        "EGU Forward (output: [B,T,H*dir], hy: [layers*dir,B,H])",
        py::arg("input"), py::arg("hx"),
        py::arg("weight_ih_list"), py::arg("weight_hh_list"),
        py::arg("bias_ih_list") = std::vector<torch::Tensor>(),
        py::arg("bias_hh_list") = std::vector<torch::Tensor>(),
        py::arg("num_layers") = 1, py::arg("dropout_p") = 0.0,
        py::arg("training") = true, py::arg("bidirectional") = false,
        py::arg("batch_first") = false
    );
}