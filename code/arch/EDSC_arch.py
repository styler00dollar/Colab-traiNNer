"""
26-Sep-21
https://github.com/Xianhang/EDSC-pytorch/blob/master/networks/EDSC.py
https://github.com/Xianhang/EDSC-pytorch/blob/master/networks/dsepconv.py
"""

import torch
import torch.nn as nn
import math

# from networks import dsepconv


import cupy
import re


class Stream:
    ptr = torch.cuda.current_stream().cuda_stream


# end

kernel_DSepconv_updateOutput = """
	extern "C" __global__ void kernel_DSepconv_updateOutput(
		const int n,
		const float* input,
		const float* vertical,
		const float* horizontal,
		const float* offset_x,
		const float* offset_y,
		const float* mask,
		float* output
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float dblOutput = 0.0;

		const int intSample = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
		const int intDepth  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
		const int intY      = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
		const int intX      = ( intIndex                                                    ) % SIZE_3(output);
		

		for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1) {
			for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1) {
			    float delta_x = OFFSET_4(offset_y, intSample, intFilterY*SIZE_1(vertical) + intFilterX, intY, intX);
			    float delta_y = OFFSET_4(offset_x, intSample, intFilterY*SIZE_1(vertical) + intFilterX, intY, intX);
			    
			    float position_x = delta_x + intX + intFilterX - (SIZE_1(horizontal) - 1) / 2 + 1;
			    float position_y = delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1;
			    if (position_x < 0)
			        position_x = 0;
			    if (position_x > SIZE_3(input) - 1)
			        position_x = SIZE_3(input) - 1;
			    if (position_y < 0)
			        position_y = 0;
			    if (position_y > SIZE_2(input) - 1)
			        position_y =  SIZE_2(input) - 1;
			    
			    int left = floor(delta_x + intX + intFilterX - (SIZE_1(horizontal) - 1) / 2 + 1);
			    int right = left + 1;
			    if (left < 0)
			        left = 0;
			    if (left > SIZE_3(input) - 1)
			        left = SIZE_3(input) - 1;
			    if (right < 0)
			        right = 0;
			    if (right > SIZE_3(input) - 1)
			        right = SIZE_3(input) - 1;
			    
			    int top = floor(delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1);
			    int bottom = top + 1;
			    if (top < 0)
			        top = 0;
			    if (top > SIZE_2(input) - 1)
			        top =  SIZE_2(input) - 1;
			    if (bottom < 0)
			        bottom = 0;   
			    if (bottom > SIZE_2(input) - 1)
			        bottom = SIZE_2(input) - 1;
			    
			    float floatValue = VALUE_4(input, intSample, intDepth, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			                       VALUE_4(input, intSample, intDepth, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			                       VALUE_4(input, intSample, intDepth, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			                       VALUE_4(input, intSample, intDepth, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
			                       
				dblOutput += floatValue * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX) * VALUE_4(mask, intSample, SIZE_1(vertical)*intFilterY + intFilterX, intY, intX);
			}
		}
		output[intIndex] = dblOutput;
	} }
"""

kernel_DSepconv_updateGradVertical = """
	extern "C" __global__ void kernel_DSepconv_updateGradVertical(
		const int n,
		const float* gradLoss,
		const float* input,
		const float* horizontal,
		const float* offset_x,
		const float* offset_y,
		const float* mask,
		float* gradVertical
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float floatOutput = 0.0;

		const int intSample   = ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical) / SIZE_1(gradVertical) ) % SIZE_0(gradVertical);
		const int intFilterY  = ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical)                        ) % SIZE_1(gradVertical);
		const int intY        = ( intIndex / SIZE_3(gradVertical)                                               ) % SIZE_2(gradVertical);
		const int intX        = ( intIndex                                                                      ) % SIZE_3(gradVertical);

		for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1){
		    int intDepth = intFilterY * SIZE_1(horizontal) + intFilterX;
		    float delta_x = OFFSET_4(offset_y, intSample, intDepth, intY, intX);
			float delta_y = OFFSET_4(offset_x, intSample, intDepth, intY, intX);
			
			float position_x = delta_x + intX + intFilterX - (SIZE_1(horizontal) - 1) / 2 + 1;
			float position_y = delta_y + intY + intFilterY - (SIZE_1(horizontal) - 1) / 2 + 1;
			if (position_x < 0)
			    position_x = 0;
			if (position_x > SIZE_3(input) - 1)
			    position_x = SIZE_3(input) - 1;
			if (position_y < 0)
			    position_y = 0;
			if (position_y > SIZE_2(input) - 1)
			    position_y =  SIZE_2(input) - 1;
		
			int left = floor(delta_x + intX + intFilterX - (SIZE_1(horizontal) - 1) / 2 + 1);
			int right = left + 1;
			if (left < 0)
			    left = 0;
			if (left > SIZE_3(input) - 1)
			    left = SIZE_3(input) - 1;
			if (right < 0)
			    right = 0;
			if (right > SIZE_3(input) - 1)
			    right = SIZE_3(input) - 1;

			int top = floor(delta_y + intY + intFilterY - (SIZE_1(horizontal) - 1) / 2 + 1);
			int bottom = top + 1;
			if (top < 0)
			    top = 0;
			if (top > SIZE_2(input) - 1)
			    top =  SIZE_2(input) - 1;
			if (bottom < 0)
			    bottom = 0;   
			if (bottom > SIZE_2(input) - 1)
			    bottom = SIZE_2(input) - 1;
			
			float floatSampled0 = VALUE_4(input, intSample, 0, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 0, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 0, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, 0, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
			float floatSampled1 = VALUE_4(input, intSample, 1, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 1, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 1, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, 1, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
			float floatSampled2 = VALUE_4(input, intSample, 2, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 2, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 2, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, 2, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
			
			floatOutput += VALUE_4(gradLoss, intSample, 0, intY, intX) * floatSampled0 * VALUE_4(horizontal, intSample, intFilterX, intY, intX) * VALUE_4(mask, intSample, intDepth, intY, intX) +
				       VALUE_4(gradLoss, intSample, 1, intY, intX) * floatSampled1 * VALUE_4(horizontal, intSample, intFilterX, intY, intX) * VALUE_4(mask, intSample, intDepth, intY, intX) +
				       VALUE_4(gradLoss, intSample, 2, intY, intX) * floatSampled2 * VALUE_4(horizontal, intSample, intFilterX, intY, intX) * VALUE_4(mask, intSample, intDepth, intY, intX);
		}
		gradVertical[intIndex] = floatOutput;
	} }

"""

kernel_DSepconv_updateGradHorizontal = """
	extern "C" __global__ void kernel_DSepconv_updateGradHorizontal(
		const int n,
		const float* gradLoss,
		const float* input,
		const float* vertical,
		const float* offset_x,
		const float* offset_y,
		const float* mask,
		float* gradHorizontal
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float floatOutput = 0.0;

		const int intSample   = ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal) / SIZE_1(gradHorizontal) ) % SIZE_0(gradHorizontal);
		const int intFilterX  = ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal)                          ) % SIZE_1(gradHorizontal);
		const int intY        = ( intIndex / SIZE_3(gradHorizontal)                                                   ) % SIZE_2(gradHorizontal);
		const int intX        = ( intIndex                                                                            ) % SIZE_3(gradHorizontal);

		for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1){
		    int intDepth = intFilterY * SIZE_1(vertical) + intFilterX;
		    float delta_x = OFFSET_4(offset_y, intSample, intDepth, intY, intX);
			float delta_y = OFFSET_4(offset_x, intSample, intDepth, intY, intX);
		
			float position_x = delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1;
			float position_y = delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1;
			if (position_x < 0)
			    position_x = 0;
			if (position_x > SIZE_3(input) - 1)
			    position_x = SIZE_3(input) - 1;
			if (position_y < 0)
			    position_y = 0;
			if (position_y > SIZE_2(input) - 1)
			    position_y =  SIZE_2(input) - 1;
		
			int left = floor(delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1);
			int right = left + 1;
			if (left < 0)
			    left = 0;
			if (left > SIZE_3(input) - 1)
			    left = SIZE_3(input) - 1;
			if (right < 0)
			    right = 0;
			if (right > SIZE_3(input) - 1)
			    right = SIZE_3(input) - 1;

			int top = floor(delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1);
			int bottom = top + 1;
			if (top < 0)
			    top = 0;
			if (top > SIZE_2(input) - 1)
			    top =  SIZE_2(input) - 1;
			if (bottom < 0)
			    bottom = 0;   
			if (bottom > SIZE_2(input) - 1)
			    bottom = SIZE_2(input) - 1;
			
			float floatSampled0 = VALUE_4(input, intSample, 0, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 0, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 0, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, 0, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
			float floatSampled1 = VALUE_4(input, intSample, 1, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 1, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 1, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, 1, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
			float floatSampled2 = VALUE_4(input, intSample, 2, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 2, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, 2, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, 2, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y));
				
			floatOutput += VALUE_4(gradLoss, intSample, 0, intY, intX) * floatSampled0 * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(mask, intSample, intDepth, intY, intX) +
				       VALUE_4(gradLoss, intSample, 1, intY, intX) * floatSampled1 * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(mask, intSample, intDepth, intY, intX) +
				       VALUE_4(gradLoss, intSample, 2, intY, intX) * floatSampled2 * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(mask, intSample, intDepth, intY, intX);
		}
		gradHorizontal[intIndex] = floatOutput;
	} }
"""

kernel_DSepconv_updateGradMask = """
	extern "C" __global__ void kernel_DSepconv_updateGradMask(
		const int n,
		const float* gradLoss,
		const float* input,
		const float* vertical,
		const float* horizontal,
		const float* offset_x,
		const float* offset_y,
		float* gradMask
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	    float floatOutput = 0.0;

		const int intSample   = ( intIndex / SIZE_3(gradMask) / SIZE_2(gradMask) / SIZE_1(gradMask) ) % SIZE_0(gradMask);
		const int intDepth    = ( intIndex / SIZE_3(gradMask) / SIZE_2(gradMask)                    ) % SIZE_1(gradMask);
		const int intY        = ( intIndex / SIZE_3(gradMask)                                       ) % SIZE_2(gradMask);
		const int intX        = ( intIndex                                                          ) % SIZE_3(gradMask);
		
		int intFilterY = intDepth / SIZE_1(vertical);
        int intFilterX = intDepth % SIZE_1(vertical);
        
        float delta_x = OFFSET_4(offset_y, intSample, intDepth, intY, intX);
		float delta_y = OFFSET_4(offset_x, intSample, intDepth, intY, intX);
		
		float position_x = delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1;
		float position_y = delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1;
		if (position_x < 0)
			position_x = 0;
		if (position_x > SIZE_3(input) - 1)
			position_x = SIZE_3(input) - 1;
		if (position_y < 0)
			position_y = 0;
		if (position_y > SIZE_2(input) - 1)
			position_y =  SIZE_2(input) - 1;
		
		int left = floor(delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1);
		int right = left + 1;
		if (left < 0)
			left = 0;
		if (left > SIZE_3(input) - 1)
			left = SIZE_3(input) - 1;
		if (right < 0)
			right = 0;
		if (right > SIZE_3(input) - 1)
			right = SIZE_3(input) - 1;

		int top = floor(delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1);
		int bottom = top + 1;
		if (top < 0)
			top = 0;
		if (top > SIZE_2(input) - 1)
			top =  SIZE_2(input) - 1;
		if (bottom < 0)
			bottom = 0;   
		if (bottom > SIZE_2(input) - 1)
			bottom = SIZE_2(input) - 1;
		
		for (int intChannel = 0; intChannel < 3; intChannel++){
		    floatOutput += VALUE_4(gradLoss, intSample, intChannel, intY, intX) * (
		                   VALUE_4(input, intSample, intChannel, top, left) * (1 + (left - position_x)) * (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, intChannel, top, right) * (1 - (right - position_x)) *  (1 + (top - position_y)) + 
			               VALUE_4(input, intSample, intChannel, bottom, left) * (1 + (left - position_x)) * (1 - (bottom - position_y)) + 
			               VALUE_4(input, intSample, intChannel, bottom, right) * (1 - (right - position_x)) * (1 - (bottom - position_y))
		                   ) * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX);
		} 
		gradMask[intIndex] = floatOutput;
	} }
"""

kernel_DSepconv_updateGradOffsetX = """
	extern "C" __global__ void kernel_DSepconv_updateGradOffsetX(
		const int n,
		const float* gradLoss,
		const float* input,
		const float* vertical,
		const float* horizontal,
		const float* offset_x,
		const float* offset_y,
		const float* mask,
		float* gradOffsetX
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	    float floatOutput = 0.0;

		const int intSample   = ( intIndex / SIZE_3(gradOffsetX) / SIZE_2(gradOffsetX) / SIZE_1(gradOffsetX) ) % SIZE_0(gradOffsetX);
		const int intDepth    = ( intIndex / SIZE_3(gradOffsetX) / SIZE_2(gradOffsetX)                       ) % SIZE_1(gradOffsetX);
		const int intY        = ( intIndex / SIZE_3(gradOffsetX)                                             ) % SIZE_2(gradOffsetX);
		const int intX        = ( intIndex                                                                   ) % SIZE_3(gradOffsetX);

		int intFilterY = intDepth / SIZE_1(vertical);
        int intFilterX = intDepth % SIZE_1(vertical);

        float delta_x = OFFSET_4(offset_y, intSample, intDepth, intY, intX);
		float delta_y = OFFSET_4(offset_x, intSample, intDepth, intY, intX);

		float position_x = delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1;
		float position_y = delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1;
		if (position_x < 0)
			position_x = 0;
		if (position_x > SIZE_3(input) - 1)
			position_x = SIZE_3(input) - 1;
		if (position_y < 0)
			position_y = 0;
		if (position_y > SIZE_2(input) - 1)
			position_y =  SIZE_2(input) - 1;
		
		int left = floor(delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1);
		int right = left + 1;
		if (left < 0)
			left = 0;
		if (left > SIZE_3(input) - 1)
			left = SIZE_3(input) - 1;
		if (right < 0)
			right = 0;
		if (right > SIZE_3(input) - 1)
			right = SIZE_3(input) - 1;

		int top = floor(delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1);
		int bottom = top + 1;
		if (top < 0)
			top = 0;
		if (top > SIZE_2(input) - 1)
			top =  SIZE_2(input) - 1;
		if (bottom < 0)
			bottom = 0;   
		if (bottom > SIZE_2(input) - 1)
			bottom = SIZE_2(input) - 1;

		for (int intChannel = 0; intChannel < 3; intChannel++){
			floatOutput += VALUE_4(gradLoss, intSample, intChannel, intY, intX) * (
		                   - VALUE_4(input, intSample, intChannel, top, left)  * (1 + (left - position_x))
		                   - VALUE_4(input, intSample, intChannel, top, right)  *  (1 - (right - position_x))
			               + VALUE_4(input, intSample, intChannel, bottom, left) * (1 + (left - position_x))
			               + VALUE_4(input, intSample, intChannel, bottom, right) * (1 - (right - position_x))
			               )
		                   * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX)
		                   * VALUE_4(mask, intSample, intDepth, intY, intX);
		} 
		gradOffsetX[intIndex] = floatOutput;
	} }
"""

kernel_DSepconv_updateGradOffsetY = """
	extern "C" __global__ void kernel_DSepconv_updateGradOffsetY(
		const int n,
		const float* gradLoss,
		const float* input,
		const float* vertical,
		const float* horizontal,
		const float* offset_x,
		const float* offset_y,
		const float* mask,
		float* gradOffsetY
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	    float floatOutput = 0.0;

		const int intSample   = ( intIndex / SIZE_3(gradOffsetX) / SIZE_2(gradOffsetX) / SIZE_1(gradOffsetX) ) % SIZE_0(gradOffsetX);
		const int intDepth    = ( intIndex / SIZE_3(gradOffsetX) / SIZE_2(gradOffsetX)                       ) % SIZE_1(gradOffsetX);
		const int intY        = ( intIndex / SIZE_3(gradOffsetX)                                             ) % SIZE_2(gradOffsetX);
		const int intX        = ( intIndex                                                                   ) % SIZE_3(gradOffsetX);

		int intFilterY = intDepth / SIZE_1(vertical);
        int intFilterX = intDepth % SIZE_1(vertical);

        float delta_x = OFFSET_4(offset_y, intSample, intDepth, intY, intX);
		float delta_y = OFFSET_4(offset_x, intSample, intDepth, intY, intX);

		float position_x = delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1;
		float position_y = delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1;
		if (position_x < 0)
			position_x = 0;
		if (position_x > SIZE_3(input) - 1)
			position_x = SIZE_3(input) - 1;
		if (position_y < 0)
			position_y = 0;
		if (position_y > SIZE_2(input) - 1)
			position_y =  SIZE_2(input) - 1;
		
		int left = floor(delta_x + intX + intFilterX - (SIZE_1(vertical) - 1) / 2 + 1);
		int right = left + 1;
		if (left < 0)
			left = 0;
		if (left > SIZE_3(input) - 1)
			left = SIZE_3(input) - 1;
		if (right < 0)
			right = 0;
		if (right > SIZE_3(input) - 1)
			right = SIZE_3(input) - 1;

		int top = floor(delta_y + intY + intFilterY - (SIZE_1(vertical) - 1) / 2 + 1);
		int bottom = top + 1;
		if (top < 0)
			top = 0;
		if (top > SIZE_2(input) - 1)
			top =  SIZE_2(input) - 1;
		if (bottom < 0)
			bottom = 0;   
		if (bottom > SIZE_2(input) - 1)
			bottom = SIZE_2(input) - 1;

		for (int intChannel = 0; intChannel < 3; intChannel++){
		    floatOutput += VALUE_4(gradLoss, intSample, intChannel, intY, intX) * (
		                   - VALUE_4(input, intSample, intChannel, top, left)  * (1 + (top - position_y)) 
		                   + VALUE_4(input, intSample, intChannel, top, right)  *  (1 + (top - position_y)) 
			               - VALUE_4(input, intSample, intChannel, bottom, left) * (1 - (bottom - position_y)) 
			               + VALUE_4(input, intSample, intChannel, bottom, right) * (1 - (bottom - position_y))
			               )
		                   * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX)
		                   * VALUE_4(mask, intSample, intDepth, intY, intX);
		} 
		gradOffsetY[intIndex] = floatOutput;
	} }
"""


def cupy_kernel(strFunction, objectVariables):
    strKernel = globals()[strFunction]

    while True:
        objectMatch = re.search("(SIZE_)([0-4])(\()([^\)]*)(\))", strKernel)

        if objectMatch is None:
            break
        # end

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objectMatch = re.search("(VALUE_)([0-4])(\()([^\)]+)(\))", strKernel)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(",")

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = [
            "(("
            + strArgs[intArg + 1].replace("{", "(").replace("}", ")").strip()
            + ")*"
            + str(intStrides[intArg])
            + ")"
            for intArg in range(intArgs)
        ]

        strKernel = strKernel.replace(
            objectMatch.group(0), strTensor + "[" + str.join("+", strIndex) + "]"
        )
    # end

    while True:
        objectMatch = re.search("(OFFSET_)([0-4])(\()([^\)]+)(\))", strKernel)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(",")

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = [
            "(("
            + strArgs[intArg + 1].replace("{", "(").replace("}", ")").strip()
            + ")*"
            + str(intStrides[intArg])
            + ")"
            for intArg in range(intArgs)
        ]

        strKernel = strKernel.replace(
            objectMatch.group(0), strTensor + "[" + str.join("+", strIndex) + "]"
        )
    # end

    return strKernel


# end


@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)


# end


class _FunctionDSepconv(torch.autograd.Function):
    @staticmethod
    def forward(self, input, vertical, horizontal, offset_x, offset_y, mask):
        self.save_for_backward(input, vertical, horizontal, offset_x, offset_y, mask)

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        assert intInputHeight == intOutputHeight + intFilterSize - 1
        assert intInputWidth == intOutputWidth + intFilterSize - 1

        assert input.is_contiguous() is True
        assert vertical.is_contiguous() is True
        assert horizontal.is_contiguous() is True
        assert offset_x.is_contiguous() is True
        assert offset_y.is_contiguous() is True
        assert mask.is_contiguous() is True

        output = input.new_zeros(
            [intSample, intInputDepth, intOutputHeight, intOutputWidth]
        )

        if input.is_cuda is True:
            n = output.nelement()
            cupy_launch(
                "kernel_DSepconv_updateOutput",
                cupy_kernel(
                    "kernel_DSepconv_updateOutput",
                    {
                        "input": input,
                        "vertical": vertical,
                        "horizontal": horizontal,
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                        "mask": mask,
                        "output": output,
                    },
                ),
            )(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n,
                    input.data_ptr(),
                    vertical.data_ptr(),
                    horizontal.data_ptr(),
                    offset_x.data_ptr(),
                    offset_y.data_ptr(),
                    mask.data_ptr(),
                    output.data_ptr(),
                ],
                stream=Stream,
            )

        elif input.is_cuda is False:
            raise NotImplementedError()

        # end

        return output

    # end

    @staticmethod
    def backward(self, gradOutput):
        input, vertical, horizontal, offset_x, offset_y, mask = self.saved_tensors

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        assert intInputHeight == intOutputHeight + intFilterSize - 1
        assert intInputWidth == intOutputWidth + intFilterSize - 1

        assert gradOutput.is_contiguous() is True

        gradInput = (
            input.new_zeros([intSample, intInputDepth, intInputHeight, intInputWidth])
            if self.needs_input_grad[0] is True
            else None
        )
        gradVertical = (
            input.new_zeros([intSample, intFilterSize, intOutputHeight, intOutputWidth])
            if self.needs_input_grad[1] is True
            else None
        )
        gradHorizontal = (
            input.new_zeros([intSample, intFilterSize, intOutputHeight, intOutputWidth])
            if self.needs_input_grad[2] is True
            else None
        )
        gradOffsetX = (
            input.new_zeros(
                [
                    intSample,
                    intFilterSize * intFilterSize,
                    intOutputHeight,
                    intOutputWidth,
                ]
            )
            if self.needs_input_grad[3] is True
            else None
        )
        gradOffsetY = (
            input.new_zeros(
                [
                    intSample,
                    intFilterSize * intFilterSize,
                    intOutputHeight,
                    intOutputWidth,
                ]
            )
            if self.needs_input_grad[4] is True
            else None
        )
        gradMask = (
            input.new_zeros(
                [
                    intSample,
                    intFilterSize * intFilterSize,
                    intOutputHeight,
                    intOutputWidth,
                ]
            )
            if self.needs_input_grad[5] is True
            else None
        )

        if input.is_cuda is True:
            nv = gradVertical.nelement()
            cupy_launch(
                "kernel_DSepconv_updateGradVertical",
                cupy_kernel(
                    "kernel_DSepconv_updateGradVertical",
                    {
                        "gradLoss": gradOutput,
                        "input": input,
                        "horizontal": horizontal,
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                        "mask": mask,
                        "gradVertical": gradVertical,
                    },
                ),
            )(
                grid=tuple([int((nv + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    nv,
                    gradOutput.data_ptr(),
                    input.data_ptr(),
                    horizontal.data_ptr(),
                    offset_x.data_ptr(),
                    offset_y.data_ptr(),
                    mask.data_ptr(),
                    gradVertical.data_ptr(),
                ],
                stream=Stream,
            )

            nh = gradHorizontal.nelement()
            cupy_launch(
                "kernel_DSepconv_updateGradHorizontal",
                cupy_kernel(
                    "kernel_DSepconv_updateGradHorizontal",
                    {
                        "gradLoss": gradOutput,
                        "input": input,
                        "vertical": vertical,
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                        "mask": mask,
                        "gradHorizontal": gradHorizontal,
                    },
                ),
            )(
                grid=tuple([int((nh + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    nh,
                    gradOutput.data_ptr(),
                    input.data_ptr(),
                    vertical.data_ptr(),
                    offset_x.data_ptr(),
                    offset_y.data_ptr(),
                    mask.data_ptr(),
                    gradHorizontal.data_ptr(),
                ],
                stream=Stream,
            )

            nx = gradOffsetX.nelement()
            cupy_launch(
                "kernel_DSepconv_updateGradOffsetX",
                cupy_kernel(
                    "kernel_DSepconv_updateGradOffsetX",
                    {
                        "gradLoss": gradOutput,
                        "input": input,
                        "vertical": vertical,
                        "horizontal": horizontal,
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                        "mask": mask,
                        "gradOffsetX": gradOffsetX,
                    },
                ),
            )(
                grid=tuple([int((nx + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    nx,
                    gradOutput.data_ptr(),
                    input.data_ptr(),
                    vertical.data_ptr(),
                    horizontal.data_ptr(),
                    offset_x.data_ptr(),
                    offset_y.data_ptr(),
                    mask.data_ptr(),
                    gradOffsetX.data_ptr(),
                ],
                stream=Stream,
            )

            ny = gradOffsetY.nelement()
            cupy_launch(
                "kernel_DSepconv_updateGradOffsetY",
                cupy_kernel(
                    "kernel_DSepconv_updateGradOffsetY",
                    {
                        "gradLoss": gradOutput,
                        "input": input,
                        "vertical": vertical,
                        "horizontal": horizontal,
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                        "mask": mask,
                        "gradOffsetX": gradOffsetY,
                    },
                ),
            )(
                grid=tuple([int((ny + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    ny,
                    gradOutput.data_ptr(),
                    input.data_ptr(),
                    vertical.data_ptr(),
                    horizontal.data_ptr(),
                    offset_x.data_ptr(),
                    offset_y.data_ptr(),
                    mask.data_ptr(),
                    gradOffsetY.data_ptr(),
                ],
                stream=Stream,
            )

            nm = gradMask.nelement()
            cupy_launch(
                "kernel_DSepconv_updateGradMask",
                cupy_kernel(
                    "kernel_DSepconv_updateGradMask",
                    {
                        "gradLoss": gradOutput,
                        "input": input,
                        "vertical": vertical,
                        "horizontal": horizontal,
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                        "gradMask": gradMask,
                    },
                ),
            )(
                grid=tuple([int((nm + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    nm,
                    gradOutput.data_ptr(),
                    input.data_ptr(),
                    vertical.data_ptr(),
                    horizontal.data_ptr(),
                    offset_x.data_ptr(),
                    offset_y.data_ptr(),
                    gradMask.data_ptr(),
                ],
                stream=Stream,
            )

        elif input.is_cuda is False:
            raise NotImplementedError()

        # end

        return (
            gradInput,
            gradVertical,
            gradHorizontal,
            gradOffsetX,
            gradOffsetY,
            gradMask,
        )


# end
# end


def FunctionDSepconv(
    tensorInput,
    tensorVertical,
    tensorHorizontal,
    tensorOffsetX,
    tensorOffsetY,
    tensorMask,
):
    return _FunctionDSepconv.apply(
        tensorInput,
        tensorVertical,
        tensorHorizontal,
        tensorOffsetX,
        tensorOffsetY,
        tensorMask,
    )


# end


class ModuleDSepconv(torch.nn.Module):
    def __init__(self):
        super(ModuleDSepconv, self).__init__()

    # end

    def forward(
        self,
        tensorInput,
        tensorVertical,
        tensorHorizontal,
        tensorOffsetX,
        tensorOffsetY,
        tensorMask,
    ):
        return _FunctionDSepconv.apply(
            tensorInput,
            tensorVertical,
            tensorHorizontal,
            tensorOffsetX,
            tensorOffsetY,
            tensorMask,
        )


# end
# end

# float floatValue = VALUE_4(input, intSample, intDepth, top, left) * (1 - (delta_x - floor(delta_x))) * (1 - (delta_y - floor(delta_y))) +
# 			                       VALUE_4(input, intSample, intDepth, top, right) * (delta_x - floor(delta_x)) *  (1 - (delta_y - floor(delta_y))) +
# 			                       VALUE_4(input, intSample, intDepth, bottom, left) * (1 - (delta_x - floor(delta_x))) * (delta_y - floor(delta_y)) +
# 			                       VALUE_4(input, intSample, intDepth, bottom, right) * (delta_x - floor(delta_x)) * (delta_y - floor(delta_y));


class Efficient_HetConv2d(nn.Module):
    def __init__(self, in_feats, out_feats, p=4, ks=3, pad=1):
        super(Efficient_HetConv2d, self).__init__()
        if in_feats % p != 0:
            raise ValueError("in_channels must be divisible by p")
        if out_feats % p != 0:
            raise ValueError("out_channels must be divisible by p")
        self.conv3x3 = nn.Conv2d(
            in_feats, out_feats, kernel_size=ks, padding=pad, groups=p
        )
        self.conv1x1_ = nn.Conv2d(in_feats, out_feats, kernel_size=1, groups=p)
        self.conv1x1 = nn.Conv2d(in_feats, out_feats, kernel_size=1)

    def forward(self, x):
        return self.conv3x3(x) + self.conv1x1(x) - self.conv1x1_(x)


class Network(nn.Module):
    """
    Network of EDSC
    """

    def __init__(self, Generated_ks=5, Het_p=4, useBias=True, isMultiple=False):
        super(Network, self).__init__()
        self.het_p = Het_p
        self.generated_ks = Generated_ks
        self.useBias = useBias
        self.isMultiple = isMultiple
        if self.isMultiple:
            self.estimator_in = 65
        else:
            self.estimator_in = 64

        def Basic(intInput, intOutput, ks, pad):
            return torch.nn.Sequential(
                Efficient_HetConv2d(
                    in_feats=intInput, out_feats=intOutput, p=self.het_p, ks=ks, pad=pad
                ),
                torch.nn.ReLU(inplace=False),
                Efficient_HetConv2d(
                    in_feats=intOutput,
                    out_feats=intOutput,
                    p=self.het_p,
                    ks=ks,
                    pad=pad,
                ),
                torch.nn.ReLU(inplace=False),
                Efficient_HetConv2d(
                    in_feats=intOutput,
                    out_feats=intOutput,
                    p=self.het_p,
                    ks=ks,
                    pad=pad,
                ),
                torch.nn.ReLU(inplace=False),
            )

        # end

        def Upsample(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                Efficient_HetConv2d(
                    in_feats=intOutput, out_feats=intOutput, p=self.het_p, ks=3, pad=1
                ),
                torch.nn.ReLU(inplace=False),
            )

        # end

        def KernelNet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.estimator_in,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=32,
                    out_channels=self.generated_ks,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                torch.nn.Conv2d(
                    in_channels=self.generated_ks,
                    out_channels=self.generated_ks,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )

        # end

        def Offsetnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.estimator_in,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=32,
                    out_channels=self.generated_ks**2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                torch.nn.Conv2d(
                    in_channels=self.generated_ks**2,
                    out_channels=self.generated_ks**2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )

        def Masknet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.estimator_in,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=32,
                    out_channels=self.generated_ks**2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                torch.nn.Conv2d(
                    in_channels=self.generated_ks**2,
                    out_channels=self.generated_ks**2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.Sigmoid(),
            )

        def Biasnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                torch.nn.Conv2d(
                    in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
                ),
            )

        self.moduleConv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            Efficient_HetConv2d(in_feats=32, out_feats=32, p=self.het_p, ks=3, pad=1),
            torch.nn.ReLU(inplace=False),
            Efficient_HetConv2d(in_feats=32, out_feats=32, p=self.het_p, ks=3, pad=1),
            torch.nn.ReLU(inplace=False),
        )

        self.moduleConv2 = Basic(32, 64, 3, 1)
        self.moduleConv3 = Basic(64, 128, 3, 1)
        self.moduleConv4 = Basic(128, 256, 3, 1)
        self.moduleConv5 = Basic(256, 512, 3, 1)

        self.moduleDeconv5 = Basic(512, 512, 3, 1)
        self.moduleDeconv4 = Basic(512, 256, 3, 1)
        self.moduleDeconv3 = Basic(256, 128, 3, 1)
        self.moduleDeconv2 = Basic(128, 64, 3, 1)

        self.moduleUpsample5 = Upsample(512, 512)
        self.moduleUpsample4 = Upsample(256, 256)
        self.moduleUpsample3 = Upsample(128, 128)
        self.moduleUpsample2 = Upsample(64, 64)

        self.moduleVertical1 = KernelNet()
        self.moduleVertical2 = KernelNet()
        self.moduleHorizontal1 = KernelNet()
        self.moduleHorizontal2 = KernelNet()

        self.moduleOffset1x = Offsetnet()
        self.moduleOffset1y = Offsetnet()
        self.moduleOffset2x = Offsetnet()
        self.moduleOffset2y = Offsetnet()

        self.moduleMask1 = Masknet()
        self.moduleMask2 = Masknet()

        self.moduleBias = Biasnet()
        # self.moduleOffset1x.register_backward_hook(self._set_lr)

    # end
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, tensorFirst, tensorSecond):
        # tensorFirst = tensors[0]
        # tensorSecond = tensors[1]
        if self.isMultiple:
            tensorTime = tensors[2]

        tensorConv1 = self.moduleConv1(torch.cat([tensorFirst, tensorSecond], 1))
        tensorConv2 = self.moduleConv2(
            torch.nn.functional.avg_pool2d(input=tensorConv1, kernel_size=2, stride=2)
        )
        tensorConv3 = self.moduleConv3(
            torch.nn.functional.avg_pool2d(input=tensorConv2, kernel_size=2, stride=2)
        )
        tensorConv4 = self.moduleConv4(
            torch.nn.functional.avg_pool2d(input=tensorConv3, kernel_size=2, stride=2)
        )
        tensorConv5 = self.moduleConv5(
            torch.nn.functional.avg_pool2d(input=tensorConv4, kernel_size=2, stride=2)
        )

        tensorDeconv5 = self.moduleUpsample5(
            self.moduleDeconv5(
                torch.nn.functional.avg_pool2d(
                    input=tensorConv5, kernel_size=2, stride=2
                )
            )
        )
        tensorDeconv4 = self.moduleUpsample4(
            self.moduleDeconv4(tensorDeconv5 + tensorConv5)
        )
        tensorDeconv3 = self.moduleUpsample3(
            self.moduleDeconv3(tensorDeconv4 + tensorConv4)
        )
        tensorDeconv2 = self.moduleUpsample2(
            self.moduleDeconv2(tensorDeconv3 + tensorConv3)
        )

        tensorCombine = tensorDeconv2 + tensorConv2

        tensorFirst = torch.nn.functional.pad(
            input=tensorFirst,
            pad=[
                int(math.floor(5 / 2.0)),
                int(math.floor(5 / 2.0)),
                int(math.floor(5 / 2.0)),
                int(math.floor(5 / 2.0)),
            ],
            mode="replicate",
        )
        tensorSecond = torch.nn.functional.pad(
            input=tensorSecond,
            pad=[
                int(math.floor(5 / 2.0)),
                int(math.floor(5 / 2.0)),
                int(math.floor(5 / 2.0)),
                int(math.floor(5 / 2.0)),
            ],
            mode="replicate",
        )

        if self.isMultiple:
            v1 = self.moduleVertical1(torch.cat([tensorCombine, tensorTime], 1))
            v2 = self.moduleVertical2(torch.cat([tensorCombine, 1.0 - tensorTime], 1))
            h1 = self.moduleHorizontal1(torch.cat([tensorCombine, tensorTime], 1))
            h2 = self.moduleHorizontal2(torch.cat([tensorCombine, 1.0 - tensorTime], 1))

            tensorDot1 = FunctionDSepconv(
                tensorFirst,
                v1,
                h1,
                self.moduleOffset1x(torch.cat([tensorCombine, tensorTime], 1)),
                self.moduleOffset1y(torch.cat([tensorCombine, tensorTime], 1)),
                self.moduleMask1(torch.cat([tensorCombine, tensorTime], 1)),
            )
            tensorDot2 = FunctionDSepconv(
                tensorSecond,
                v2,
                h2,
                self.moduleOffset2x(torch.cat([tensorCombine, 1.0 - tensorTime], 1)),
                self.moduleOffset2y(torch.cat([tensorCombine, 1.0 - tensorTime], 1)),
                self.moduleMask2(torch.cat([tensorCombine, 1.0 - tensorTime], 1)),
            )
        else:
            v1 = self.moduleVertical1(tensorCombine)
            v2 = self.moduleVertical2(tensorCombine)
            h1 = self.moduleHorizontal1(tensorCombine)
            h2 = self.moduleHorizontal2(tensorCombine)
            offset1x = self.moduleOffset1x(tensorCombine)
            offset1y = self.moduleOffset1y(tensorCombine)
            offset2x = self.moduleOffset2x(tensorCombine)
            offset2y = self.moduleOffset2y(tensorCombine)
            mask1 = self.moduleMask1(tensorCombine)
            mask2 = self.moduleMask2(tensorCombine)

            tensorDot1 = FunctionDSepconv(
                tensorFirst, v1, h1, offset1x, offset1y, mask1
            )
            tensorDot2 = FunctionDSepconv(
                tensorSecond, v2, h2, offset2x, offset2y, mask2
            )

        if self.useBias:
            return tensorDot1 + tensorDot2 + self.moduleBias(tensorCombine)
        else:
            return tensorDot1 + tensorDot2

    # end forward


# end class
