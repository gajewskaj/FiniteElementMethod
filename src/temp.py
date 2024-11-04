# I have the below code:

import module1 as my_mod1
if not neg_cond:
    import module2 as my_mod2

def fun() -> my_mod1.x | my_mod2.x:
    if neg_cond:
        return my_mod1.x
    else:
        return my_mod2.x