/* Copyright (c) 2023 Amazon */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include "nnet.h"
#include "arch.h"
#include "nnet.h"

/* This is a bit of a hack because we need to build nnet_data.c and plc_data.c without USE_WEIGHTS_FILE,
   but USE_WEIGHTS_FILE is defined in config.h. */
#undef HAVE_CONFIG_H
#ifdef USE_WEIGHTS_FILE
#undef USE_WEIGHTS_FILE
#endif
#include "rnnoise_data.c"

void write_weights(const WeightArray *list, FILE *fout)
{
	unsigned char zeros[WEIGHT_BLOCK_SIZE] = {0};

	for(int i = 0; list[i].name != NULL; i++)
	{
		const WeightArray *arr = &list[i];
		WeightHead head;

		printf("## Writing: %s, arr->type: %d, arr->size: %d\n", arr->name, arr->type, arr->size);

		// Write magic bit
		memcpy(head.head, "DNNw", 4);
		head.version = WEIGHT_BLOB_VERSION;
		head.type = arr->type;
		head.size = arr->size;
		head.block_size = (arr->size + WEIGHT_BLOCK_SIZE - 1) / WEIGHT_BLOCK_SIZE * WEIGHT_BLOCK_SIZE;

		memset(head.name, 0, sizeof(head.name));
		strncpy(head.name, arr->name, sizeof(head.name) - 1);
		head.name[sizeof(head.name) - 1] = (char *)NULL;

		// Check if Head Size is wrong
		if(sizeof(head) != WEIGHT_BLOCK_SIZE)
		{
			printf("ERR: Header Size is Wrong!! - [%zu]\n", sizeof(head));
			continue;
		}

		// Write Header to File
		fwrite(&head, 1, WEIGHT_BLOCK_SIZE, fout);
		// Write Data to File
		fwrite(arr->data, 1, arr->size, fout);
		// Fill 0 to empty padding
		fwrite(zeros, 1, head.block_size - arr->size, fout);

		printf("## Wrote: %s, head->type: %d, head->size: %d, head->block_size: %d\n", head.name, head.type, head.size, head.block_size);
	}
}

int main(void)
{
	FILE *fout = fopen("weights_blob.bin", "wb");
	write_weights(rnnoise_arrays, fout);

	fclose(fout);
	return 0;
}
