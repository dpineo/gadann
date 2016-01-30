//
// GADANN - GPU Accelerated Deep Artificial Neural Network
//
// Copyright (C) 2014 Daniel Pineo (daniel@pineo.net)
//
// Permission is hereby granted, free of charge, to any person obtaining a 
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation 
// the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the 
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included 
// in all copies or substantial portions of the Software
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
// DEALINGS IN THE SOFTWARE.

__global__ void mult_kernel(float* a, float* b, float* out, int size) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size) {
		out[i] = a[i]*b[i];
	}
}

__global__ void div_kernel(float* a, float* b, float* out, int size) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size) {
		out[i] = a[i]/b[i];
	}
}

__global__ void sample_kernel(float* in, float* p, float* out) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (in[i] > p[i]) {
		out[i] = 1.0;
	} else {
		out[i] = 0.0;
	}
}

__global__ void broadcast_kernel(float* in, float* out) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	out[i] = in[threadIdx.x];
}

__global__ void sqrt_kernel(float* z, float* out, int size) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size) {
		out[i] = sqrt(z[i]);
	}
}

__global__ void logistic_kernel(float* z, float* out, int size) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size) {
		out[i] = 1.0/(1.0 + exp(-z[i]));
	}
}

__global__ void dlogistic_kernel(float* in, float* y, float* out, int size) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size) {
		out[i] = in[i]*y[i]*(1.0-y[i]);
	}
}



// revisit: storing to global memory + each thread is summing
__global__ void channel_sum_kernel(float* z, float* out, int blocksize) {
	int i = gridDim.y*gridDim.x*blocksize*threadIdx.x
		+ blockIdx.y*gridDim.x
		+ blockIdx.x;

	int j = gridDim.y*gridDim.x*threadIdx.x
		+ blockIdx.y*gridDim.x
		+ blockIdx.x;

	float sum = 0;
	for (int n = 0; n<blocksize; n++) {
		sum += z[i + n*gridDim.y*gridDim.x];
	}
	out[j] = sum;
}


// revisit: storing to global memory + each thread is summing
__global__ void channel_max_kernel(float* z, float* out, int blocksize) {
	int i = gridDim.y*gridDim.x*blocksize*threadIdx.x
		+ blockIdx.y*gridDim.x
		+ blockIdx.x;

	int j = gridDim.y*gridDim.x*threadIdx.x
		+ blockIdx.y*gridDim.x
		+ blockIdx.x;

	float max = z[i];
	for (int n = 1; n < blocksize; n++) {
		max = fmax(max, z[i + n*gridDim.y*gridDim.x]);
	}
	out[j] = max;
}



// revisit: storing to global memory + each thread is summing
__global__ void channel_subtract_kernel(float* z, float* max, float* out, int blocksize) {
	int i = blockIdx.z*blockDim.x*gridDim.y*gridDim.x
		+ threadIdx.x*gridDim.y*gridDim.x
		+ blockIdx.y*gridDim.x
		+ blockIdx.x;

	int j = blockIdx.z*gridDim.y*gridDim.x
		+ blockIdx.y*gridDim.x
		+ blockIdx.x;

	out[i] = exp(z[i] - max[j]);
}


__global__ void softmax_divide_kernel(float* z, float* sum, int blocksize) {
	int i = blockIdx.z*blockDim.x*gridDim.y*gridDim.x
		+ threadIdx.x*gridDim.y*gridDim.x
		+ blockIdx.y*gridDim.x
		+ blockIdx.x;

	int j = blockIdx.z*gridDim.y*gridDim.x
		+ blockIdx.y*gridDim.x
		+ blockIdx.x;

	z[i] = z[i] / sum[j];
}

/*
// channelwise max (batch, channel, y, z) axis order
__global__ void channel_max_kernel(float* z, float* out) {
	int channel_step = gridDim.y*gridDim.z;
	int i0 = blockIdx.z*channel_step + blockIdx.y*blockDim.x + blockIdx.x;
	float max = z[i0];
	for (int n = 1; n < blockDim.x; n++) {
		max = fmax(max, z[blockIdx.x*blockDim.x + n]);
	}
}
*/

// REVISIT: naive implementation
// Compute channelwise softmax
__global__ void softmax_naive_kernel(float* z, float* out) {
	// get block max
	int channel_step = gridDim.y*gridDim.z;
	int i0 = blockIdx.x*blockDim.x;
	float max = z[blockIdx.x*blockDim.x];
	for (int n = 1; n<blockDim.x; n++) {
		max = fmax(max, z[blockIdx.x*blockDim.x + n]);
	}
	__syncthreads();

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	out[i] = exp(z[i] - max);
	__syncthreads();

	float sum = 0;
	for (int n=0; n<blockDim.x; n++) {
		sum += out[blockIdx.x*blockDim.x + n];
	}
	__syncthreads();

	out[i] = out[i]/sum;
}

__global__ void dsoftmax_kernel(float* in, float* a, float* out) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	out[i] = in[i];
}

__global__ void tanh_kernel(float* z, float* out) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	out[i] = tanh(z[i]);
}

__global__ void dtanh_kernel(float* in, float* a, float* out) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	out[i] = in[i]*(1.0-a[i]*a[i]);
}

__global__ void relu_kernel(float* z, float* out, int size) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size) {
		out[i] = max(z[i], 0.0f);
	}
}

__global__ void drelu_kernel(float* in, float* a, float* out, int size) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size) {
		out[i] = in[i] > 1.0;
	}
}

__global__ void softplus_kernel(float* z, float* out) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	out[i] = log(1.0 + exp(z[i]));
}

__global__ void dsoftplus_kernel(float* in, float* a, float* out) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	out[i] = 1.0/(1.0 + exp(-in[i]));
}

__global__ void identity_kernel(float* z, float* out) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	out[i] = z[i];
}

__global__ void didentity_kernel(float* in, float* a, float* out) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	out[i] = in[i];
}

__global__ void diff_kernel(float* output, float* targets, float* result) { 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	result[i] = (output[i] - targets[i]);
}

__global__ void mosaic_orig_kernel(float* input, float* output, int grid_size, int border_size) { 
	int i = 
		  blockIdx.y *gridDim.x *blockDim.z*blockDim.y*blockDim.x
		+ blockIdx.x *blockDim.z*blockDim.y*blockDim.x
		+ threadIdx.z*blockDim.y*blockDim.x
		+ threadIdx.y*blockDim.x
		+ threadIdx.x;

	int j = 
		  blockIdx.y*(blockDim.y+border_size)*gridDim.x*(blockDim.x+border_size)*blockDim.z
		+ threadIdx.y*gridDim.x*(blockDim.x+border_size)*blockDim.z
		+ blockIdx.x *(blockDim.x+border_size)*blockDim.z
		+ threadIdx.x*blockDim.z
		+ threadIdx.z;
	
	if (blockIdx.y*gridDim.x + blockIdx.x < grid_size ) {
		output[j] = input[i];
	}
}

__global__ void mosaic_kernel(float* input, float* output, int grid_size, int border_size) { 
	int i = 
		  blockIdx.y *gridDim.x *gridDim.z*blockDim.y*blockDim.x
		+ blockIdx.x *gridDim.z*blockDim.y*blockDim.x
		+ blockIdx.z*blockDim.y*blockDim.x
		+ threadIdx.y*blockDim.x
		+ threadIdx.x;

	int j = 
		  blockIdx.y*(blockDim.y+border_size)*gridDim.x*(blockDim.x+border_size)*gridDim.z
		+ threadIdx.y*gridDim.x*(blockDim.x+border_size)*gridDim.z
		+ blockIdx.x *(blockDim.x+border_size)*gridDim.z
		+ threadIdx.x*gridDim.z
		+ blockIdx.z;
	
	if (blockIdx.y*gridDim.x + blockIdx.x < grid_size ) {
		output[j] = input[i];
	}
}


#define KERNEL_DIAMETER 16
#define KERNEL_AREA 256

#define BORDERED_STRIDE ((gridDim.x+1)*blockDim.x)
#define BORDERED_AREA ((gridDim.y+1)*blockDim.y*BORDERED_STRIDE)
#define BORDERED_ID ((blockIdx.y*blockDim.y+threadIdx.y)*BORDERED_STRIDE + blockIdx.x*blockDim.x+threadIdx.x)

#define UNBORDERED_STRIDE (gridDim.x*blockDim.x)
#define UNBORDERED_AREA (gridDim.y*blockDim.y*UNBORDERED_STRIDE)
#define UNBORDERED_ID ((blockIdx.y*blockDim.y+threadIdx.y)*UNBORDERED_STRIDE + blockIdx.x*blockDim.x+threadIdx.x)

//-------------------- conv2d_16x16_kernel ----------------------
//  Transform from input space to feature space.via a set of 16x16 weight kernels
//  This kernel assumes that the array sizes are a multiple of 16, so there is no boundary checking.
//  The expected shape is that of the upper array.

__global__ void conv2d_16x16_kernel(float* lower, float* weights, float* bias, float* upper, int lower_depth, int upper_depth) {
	int i = UNBORDERED_ID;
	int stride = UNBORDERED_STRIDE;
	int area = UNBORDERED_AREA;

	int i_lower = BORDERED_ID;
	int stride_lower = BORDERED_STRIDE;
	int area_lower = BORDERED_AREA;

	__shared__ float w[16][16];
	__shared__ float l[2*16][2*16];

	for ( int k=0; k<upper_depth; k++ ) {
		upper[k*area+i] = bias[k];
	}
	__syncthreads();

	for ( int j=0; j<lower_depth; j++ ) {
		l[threadIdx.y][threadIdx.x] = lower[j*area_lower+i_lower];
		l[threadIdx.y][threadIdx.x+16] = lower[j*area_lower+i_lower+16];
		l[threadIdx.y+16][threadIdx.x] = lower[j*area_lower+i_lower+16*stride_lower];
		l[threadIdx.y+16][threadIdx.x+16] = lower[j*area_lower+i_lower+16*stride_lower+16];

		for ( int k=0; k<upper_depth; k++ ) {
			w[threadIdx.y][threadIdx.x] = weights[(k*lower_depth+j)*256 + threadIdx.y*16 + threadIdx.x];
			__syncthreads();

			float sum = 0;
			for (int dy=0; dy<16; dy++) {
				for (int dx=0; dx<16; dx++) {
					sum += l[threadIdx.y+dy][threadIdx.x+dx] * w[dy][dx];
				}
			}
			upper[k*area+i] += sum;
			__syncthreads();
		}
	}
}


//-------------------- bconv2d_16x16_kernel ----------------------
//  Transform from the feature space to the input space.via a set of 16x16 weight kernels
//  This kernel assumes that the array sizes are a multiple of 16, so there is no boundary checking.
//  The expected shape is that of the *output* array.
//
__global__ void bconv2d_16x16_kernel(float* upper, float* weights, float* bias, float* lower, int upper_depth, int lower_depth) {
	int i = UNBORDERED_ID;
	int area = UNBORDERED_AREA;
	int stride = UNBORDERED_STRIDE;

	int i_lower = BORDERED_ID;
	int stride_lower = BORDERED_STRIDE;
	int area_lower = BORDERED_AREA;

	__shared__ float w[16][16];
	__shared__ float u[16][16];
	__shared__ float l[2*16][2*16];

	float sum = 0.0f;
	for ( int j=0; j<lower_depth; j++ ) {
		/*
		l[threadIdx.y][threadIdx.x] = bias[j];
		l[threadIdx.y][threadIdx.x+16] = bias[j];
		l[threadIdx.y+16][threadIdx.x] = bias[j];
		l[threadIdx.y+16][threadIdx.x+16] = bias[j];
		*/
		l[threadIdx.y][threadIdx.x] = 0;
		l[threadIdx.y][threadIdx.x+16] = 0;
		l[threadIdx.y+16][threadIdx.x] = 0;
		l[threadIdx.y+16][threadIdx.x+16] = 0;

		for ( int k=0; k<upper_depth; k++ ) {
			w[threadIdx.y][threadIdx.x] = weights[(k*lower_depth+j)*256 + threadIdx.y*16 + threadIdx.x];
			u[threadIdx.y][threadIdx.x] = upper[k*area+i];
			__syncthreads();
			for (int dy=0; dy<KERNEL_DIAMETER; dy++) {
				for (int dx=0; dx<KERNEL_DIAMETER; dx++) {
					atomicAdd(&l[threadIdx.y+dy][threadIdx.x+dx], u[threadIdx.y][threadIdx.x]*w[15-dy][15-dx]);
				}
			}
			__syncthreads();
		}

		atomicAdd(&lower[j*area_lower+i_lower], l[threadIdx.y][threadIdx.x]);
		atomicAdd(&lower[j*area_lower+i_lower+16], l[threadIdx.y][threadIdx.x+16]);
		atomicAdd(&lower[j*area_lower+i_lower+16*stride_lower], l[threadIdx.y+16][threadIdx.x]);
		atomicAdd(&lower[j*area_lower+i_lower+16*stride_lower+16], l[threadIdx.y+16][threadIdx.x+16]);
		__syncthreads();
	}
}

//-------------------- conv2d_16x16_kernel ----------------------
//  Transform from input space to feature space.via a set of 16x16 kernels
//  This kernel assumes that the array sizes are a multiple of 16, so there is no boundary checking.
//  The expected shape is that of the *upper* array.
__global__ void iconv2d_16x16_kernel(float* lower, float* upper, float* weights, int lower_depth, int upper_depth) {
	int i = UNBORDERED_ID;
	int stride = UNBORDERED_STRIDE;
	int area = UNBORDERED_AREA;

	int i_lower = BORDERED_ID;
	int stride_lower = BORDERED_STRIDE;
	int area_lower = BORDERED_AREA;

	__shared__ float l[2*16][2*16];
	__shared__ float u[16][16];
	
	for ( int j=0; j<lower_depth; j++ ) {
		l[threadIdx.y][threadIdx.x] = lower[j*area_lower+i_lower];
		l[threadIdx.y][threadIdx.x+16] = lower[j*area_lower+16+i_lower];
		l[threadIdx.y+16][threadIdx.x] = lower[j*area_lower+16*stride_lower+i_lower];
		l[threadIdx.y+16][threadIdx.x+16] = lower[j*area_lower+16*stride_lower+16+i_lower];

		for ( int k=0; k<upper_depth; k++ ) {
			u[threadIdx.y][threadIdx.x] = upper[k*area+i];
			__syncthreads();

			float sum = 0;
			for (int dy=0; dy<16; dy++) {
				for (int dx=0; dx<16; dx++) {
					sum += l[threadIdx.y+dy][threadIdx.x+dx] * u[dy][dx];
				}
			}
			weights[(k*lower_depth + j)*256 + threadIdx.y*16 + threadIdx.x] += sum;
			__syncthreads();
		}
	}
}


__global__ void iconv2d_v_bias_16x16_kernel(float* visible, float* v_bias, int num_in_weights) {
	int i = UNBORDERED_ID;
	int stride = UNBORDERED_STRIDE;
	int area = UNBORDERED_AREA;

	__shared__ float v[2*16][2*16];
	
	for ( int k=0; k<num_in_weights; k++ ) {
		v[threadIdx.y][threadIdx.x] = visible[i];
		v[threadIdx.y][threadIdx.x+16] = visible[i+16];
		v[threadIdx.y+16][threadIdx.x] = visible[i+16*stride];
		v[threadIdx.y+16][threadIdx.x+16] = visible[i+16*stride+16];
		__syncthreads();

		float sum = 0;
		for (int dy=0; dy<16; dy++) {
			for (int dx=0; dx<16; dx++) {
				sum += v[threadIdx.y+dy][threadIdx.x+dx];
			}
		}
		__syncthreads();
		v_bias[k*256 + threadIdx.y*16 + threadIdx.x] = sum;
	}
}


__global__ void iconv2d_h_bias_16x16_kernel(float* hidden, float* h_bias, int num_out_weights) {
	int i = UNBORDERED_ID;
	int stride = UNBORDERED_STRIDE;
	int area = UNBORDERED_AREA;

	__shared__ float h[16][16];
	for ( int p=0; p<num_out_weights; p++ ) {
		h[threadIdx.y][threadIdx.x] = hidden[i];
		__syncthreads();

		float sum = 0;
		for (int dy=0; dy<16; dy++) {
			for (int dx=0; dx<16; dx++) {
				sum += h[dy][dx];
			}
		}
		__syncthreads();
		h_bias[p] = sum;
	}
}

__global__ void iconv2d_v_bias_16x16_naive_kernel(float* visible, float* v_bias, int width, int height) {
	int i = UNBORDERED_ID;
	float sum = 0;
	for (int dy=0; dy<16; dy++) {
		for (int dx=0; dx<16; dx++) {
			sum += visible[(i/16)*32 + i%16 + dy*32 + dx];
		}
	}
	v_bias[i] = sum;
}

// REVISIT: not convolutional
__global__ void iconv2d_h_bias_16x16_naive_kernel(float* hidden, float* h_bias, int h_size) {
	int p = UNBORDERED_ID;

	float sum = 0;
	for (int i=0; i<h_size; i++) {
		sum += hidden[p*h_size + i];
	}
	h_bias[p] = sum;
}

__global__ void image_2_design_kernel(float* image, float* design) {
	int i = BORDERED_ID;
	int stride = BORDERED_STRIDE;
	int m = ((blockIdx.y*blockDim.y+threadIdx.y) * (blockDim.x*gridDim.x) + blockIdx.x*blockDim.x + threadIdx.x);

	__shared__ float in[2*KERNEL_DIAMETER][2*KERNEL_DIAMETER];
	in[threadIdx.y][threadIdx.x] = image[i];
	in[threadIdx.y][threadIdx.x+16] = image[i+16];
	in[threadIdx.y+16][threadIdx.x] = image[i+16*stride];
	in[threadIdx.y+16][threadIdx.x+16] = image[i+16*stride+16];
	__syncthreads();


	for (int dy=0; dy<KERNEL_DIAMETER; dy++) {
		for (int dx=0; dx<KERNEL_DIAMETER; dx++) {
			design[m*256 + 16*dy + dx] = in[threadIdx.y+dy][threadIdx.x+dx];
		}
	}
	__syncthreads();
}

__global__ void design_2_image_kernel(float* design, float* image) {
	int i = BORDERED_ID;
	int stride = BORDERED_STRIDE;
	int m = ((blockIdx.y*blockDim.y+threadIdx.y) * (blockDim.x*gridDim.x) + blockIdx.x*blockDim.x+threadIdx.x);

	for (int dy=0; dy<KERNEL_DIAMETER; dy++) {
		for (int dx=0; dx<KERNEL_DIAMETER; dx++) {
			atomicAdd(&image[i+(dy-8)*stride+(dx-8)], design[m*256 + 16*dy + dx]);
		}
	}
}
