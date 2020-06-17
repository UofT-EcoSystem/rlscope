//
// Created by jgleeson on 2020-06-15.
//

#pragma once

namespace rlscope {

bool env_is_on(const char* var, bool dflt, bool debug);

bool is_yes(const char* env_var, bool default_value);
bool is_no(const char* env_var, bool default_value);


} // namespace rlscope
