module attributes {
    glwe.test = #glwe.expr<((@n * ((@q ** 2. div (96.0 * ( @new_q div 2.0 ) ** 2.)) + 1.0 div 48.0)) div @q**2.0) + @input_variance>
    //glwe.symbol = #glwe.expr<@a>,
    //glwe.constant = #glwe.expr<1.>,
    //glwe.add = #glwe.expr<@a + @b>,
    //glwe.mul = #glwe.expr<@a * @b>,
    //glwe.pow = #glwe.expr<@a ** @b>,
    //glwe.div = #glwe.expr<@a div @b>,
    //glwe.max = #glwe.expr<max(@a, @b)>,
    //glwe.min = #glwe.expr<min(@a, @b)>,
    //glwe.abs = #glwe.expr<abs(@a)>,
    //glwe.floor = #glwe.expr<floor(@a)>,
    //glwe.ceil = #glwe.expr<ceil(@a)>,
    //glwe.switched = #glwe.expr<2. ** (@b * @l)>,
    //glwe.ms_variance = #glwe.expr<(@n * ((@q ** 2. div (96.0 * ( @new_q div 2.0 ) ** 2.)) + 1.0 div 48.0)) div @q**2 + @input_variance>

    //,

    //glwe.domain = #glwe.let<@a = [1.]>
} {

}