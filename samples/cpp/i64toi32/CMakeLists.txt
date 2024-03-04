# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME i64toi32
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES ${GFLAGS_TARGET})
