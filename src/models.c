/* Copyright (c) 2018 Gregor Richards */
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

#include <string.h>

#include "rnnoise-nu.h"

/* This file is just a list of the built-in models and a way of fetching them.
 * Nothing fancy. */

static const char *model_names[] = {
    "orig",
    "cb",
    "mp",
    "bd",
    "lq",
    "sh",
    NULL
};

extern const struct RNNModel
    model_orig,
    model_cb,
    model_mp,
    model_bd,
    model_lq,
    model_sh;

static const struct RNNModel *models[] = {
    &model_orig,
    &model_cb,
    &model_mp,
    &model_bd,
    &model_lq,
    &model_sh
};

const char **rnnoise_models()
{
    return model_names;
}

RNNModel *rnnoise_get_model(const char *name)
{
    int i;
    for (i = 0; model_names[i]; i++) {
        if (!strcmp(name, model_names[i]))
            return (RNNModel *) models[i];
    }
    return NULL;
}
