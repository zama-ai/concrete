// RUN: concretecompiler --action=roundtrip %s  2>&1| FileCheck %s
module attributes {
    // CHECK: glwe.e00_constant = #glwe.expr<2.100000e+00>
    // CHECK: glwe.e01_symbol = #glwe.expr<@mysymbol>
    // CHECK: glwe.e02_add = #glwe.expr<@a + @b>
    // CHECK: glwe.e04_mul = #glwe.expr<@a * @b>
    // CHECK: glwe.e06_pow = #glwe.expr<@a ** @b>
    // CHECK: glwe.e07_div = #glwe.expr<@a / @b>
    // CHECK: glwe.e08_max = #glwe.expr<max(@a, @b)>
    // CHECK: glwe.e09_min = #glwe.expr<min(@a, @b)>
    // CHECK: glwe.e11_abs = #glwe.expr<abs(@a)>
    // CHECK: glwe.e12_floor = #glwe.expr<floor(@a)>
    // CHECK: glwe.e13_ceil = #glwe.expr<ceil(@a)>
    // CHECK: glwe.ms_variance = #glwe.expr<((@n * ((@q ** 2.000000e+00) / ((9.600000e+01 * @new_q / 2.000000e+00) ** 2.000000e+00) + 1.000000e+00 / 4.800000e+01)) / @q ** 2.000000e+00) + @input_variance>,
    // CHECK: glwe.n_domain_flt = #glwe.domain<@n in[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>,
    // CHECK: glwe.n_domain_flt_intlit = #glwe.domain<@n in[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>,
    // CHECK: glwe.n_domain_int = #glwe.domain<@n in[1, 2, 3, 4, 5]>,
    // CHECK: glwe.x_constr1 = #glwe.constraint<@n + (@a * @b) gt @c>,
    // CHECK: glwe.x_constr2 = #glwe.constraint<@n + (@a * @b) lt min(@p, @c + @r)>
    glwe.e00_constant = #glwe.expr<2.1>,
    glwe.e01_symbol = #glwe.expr<@mysymbol>,
    glwe.e02_add = #glwe.expr<@a + @b>,
    glwe.e03_sub = #glwe.expr<@a - @b>,
    glwe.e04_mul = #glwe.expr<@a * @b>,
    glwe.e06_pow = #glwe.expr<@a ** @b>,
    glwe.e07_div = #glwe.expr<@a / @b>,
    glwe.e08_max = #glwe.expr<max(@a, @b)>,
    glwe.e09_min = #glwe.expr<min(@a, @b)>,
    glwe.e10_neg = #glwe.expr<- @a>,
    glwe.e11_abs = #glwe.expr<abs(@a)>,
    glwe.e12_floor = #glwe.expr<floor(@a)>,
    glwe.e13_ceil = #glwe.expr<ceil(@a)>,
    glwe.ms_variance = #glwe.expr< (@n * ((@q ** 2. / (96.0 * ( @new_q / 2.0 ) ** 2.)) + 1.0 / 48.0)) / @q**2. + @input_variance>,
    glwe.n_domain_flt = #glwe.domain<@n in [1.0, 2.0, 3.0, 4.0, 5.0]>,
    glwe.n_domain_flt_intlit = #glwe.domain<@n in [ 1, 2, 3, 4.0, 5]>,
    glwe.n_domain_int = #glwe.domain<@n in [ 1, 2, 3, 4, 5]>,
    glwe.x_constr1 = #glwe.constraint<@n + (@a * @b) gt @c>,
    glwe.x_constr2 = #glwe.constraint<@n + (@a * @b) lt min(@p, @c + @r)>
} {

}
