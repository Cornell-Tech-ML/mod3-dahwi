# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# Task 3.1: Parallelization & Task 3.2: Matrix Multiplication #
## Diagnostics Output ##

```
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (175)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (175) 
------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                       | 
        out: Storage,                                                               | 
        out_shape: Shape,                                                           | 
        out_strides: Strides,                                                       | 
        in_storage: Storage,                                                        | 
        in_shape: Shape,                                                            | 
        in_strides: Strides,                                                        | 
    ) -> None:                                                                      | 
        if np.array_equal(out_strides, in_strides) and np.array_equal(              | 
            out_shape, in_shape                                                     | 
        ):                                                                          | 
            for i in prange(len(out)):----------------------------------------------| #0
                out[i] = fn(in_storage[i])                                          | 
        else:                                                                       | 
            for i in prange(len(out)):----------------------------------------------| #1
                out_index = np.empty(MAX_DIMS, np.int32)                            | 
                in_index = np.empty(MAX_DIMS, np.int32)                             | 
                to_index(i, out_shape, out_index)                                   | 
                broadcast_index(out_index, out_shape, in_shape, in_index)           | 
                out[i] = fn(in_storage[index_to_position(in_index, in_strides)])    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (190) is hoisted out of the 
parallel loop labelled #1 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (191) is hoisted out of the 
parallel loop labelled #1 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (222)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (222) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        if (                                                               | 
            np.array_equal(out_strides, a_strides)                         | 
            and np.array_equal(out_strides, b_strides)                     | 
            and np.array_equal(out_shape, a_shape)                         | 
            and np.array_equal(out_shape, b_shape)                         | 
        ):                                                                 | 
            for i in prange(len(out)):-------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                    | 
        else:                                                              | 
            for i in prange(len(out)):-------------------------------------| #3
                out_index = np.empty(MAX_DIMS, np.int32)                   | 
                a_index = np.empty(MAX_DIMS, np.int32)                     | 
                b_index = np.empty(MAX_DIMS, np.int32)                     | 
                to_index(i, out_shape, out_index)                          | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                out[i] = fn(                                               | 
                    a_storage[index_to_position(a_index, a_strides)],      | 
                    b_storage[index_to_position(b_index, b_strides)],      | 
                )                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (243) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (244) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (245) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (278)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (278) 
-------------------------------------------------------------|loop #ID
    def _reduce(                                             | 
        out: Storage,                                        | 
        out_shape: Shape,                                    | 
        out_strides: Strides,                                | 
        a_storage: Storage,                                  | 
        a_shape: Shape,                                      | 
        a_strides: Strides,                                  | 
        reduce_dim: int,                                     | 
    ) -> None:                                               | 
        for i in prange(len(out)):---------------------------| #4
            reduce_size = a_shape[reduce_dim]                | 
            reduce_stride = a_strides[reduce_dim]            | 
            out_index = np.empty(MAX_DIMS, np.int32)         | 
            to_index(i, out_shape, out_index)                | 
            pos = index_to_position(out_index, a_strides)    | 
            acc = out[i]                                     | 
            for _ in range(reduce_size):                     | 
                acc = fn(acc, a_storage[pos])                | 
                pos += reduce_stride                         | 
            out[i] = acc                                     | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (290) is hoisted out of the 
parallel loop labelled #4 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (302)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/dahwi/MLE/mod3-dahwi/minitorch/fast_ops.py (302) 
-------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                               | 
    out: Storage,                                                                          | 
    out_shape: Shape,                                                                      | 
    out_strides: Strides,                                                                  | 
    a_storage: Storage,                                                                    | 
    a_shape: Shape,                                                                        | 
    a_strides: Strides,                                                                    | 
    b_storage: Storage,                                                                    | 
    b_shape: Shape,                                                                        | 
    b_strides: Strides,                                                                    | 
) -> None:                                                                                 | 
    """NUMBA tensor matrix multiply function.                                              | 
                                                                                           | 
    Should work for any tensor shapes that broadcast as long as                            | 
                                                                                           | 
    ```                                                                                    | 
    assert a_shape[-1] == b_shape[-2]                                                      | 
    ```                                                                                    | 
                                                                                           | 
    Optimizations:                                                                         | 
                                                                                           | 
    * Outer loop in parallel                                                               | 
    * No index buffers or function calls                                                   | 
    * Inner loop should have no global writes, 1 multiply.                                 | 
                                                                                           | 
                                                                                           | 
    Args:                                                                                  | 
    ----                                                                                   | 
        out (Storage): storage for `out` tensor                                            | 
        out_shape (Shape): shape for `out` tensor                                          | 
        out_strides (Strides): strides for `out` tensor                                    | 
        a_storage (Storage): storage for `a` tensor                                        | 
        a_shape (Shape): shape for `a` tensor                                              | 
        a_strides (Strides): strides for `a` tensor                                        | 
        b_storage (Storage): storage for `b` tensor                                        | 
        b_shape (Shape): shape for `b` tensor                                              | 
        b_strides (Strides): strides for `b` tensor                                        | 
                                                                                           | 
    Returns:                                                                               | 
    -------                                                                                | 
        None : Fills in `out`                                                              | 
                                                                                           | 
    """                                                                                    | 
    batch_size, out_rows, out_cols = out_shape                                             | 
    inner_dim = a_shape[-1]                                                                | 
                                                                                           | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                 | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                 | 
    out_batch_stride = out_strides[0] if out_shape[0] > 1 else 0                           | 
                                                                                           | 
    for n in prange(batch_size):-----------------------------------------------------------| #5
        for i in range(out_rows):                                                          | 
            for j in range(out_cols):                                                      | 
                # Compute dot product for the (i, j) element in the current batch          | 
                acc = 0.0                                                                  | 
                # Inner loop over the shared dimension (columns of `a` / rows of `b`)      | 
                for k in range(inner_dim):                                                 | 
                    # Calculate positions for `a` and `b`                                  | 
                    a_pos = n * a_batch_stride + i * a_strides[1] + k * a_strides[2]       | 
                    b_pos = n * b_batch_stride + k * b_strides[1] + j * b_strides[2]       | 
                    acc += a_storage[a_pos] * b_storage[b_pos]                             | 
                # Store the result in the output tensor                                    | 
                out[n * out_batch_stride + i * out_strides[1] + j * out_strides[2]] = (    | 
                    acc                                                                    | 
                )                                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# Task 3.4: CUDA Matrix Multiplication #
![Graph](graph.png)
```
Timing summary
Size: 64
    fast: 0.00307
    gpu: 0.00566
Size: 128
    fast: 0.01456
    gpu: 0.01317
Size: 256
    fast: 0.09175
    gpu: 0.04845
Size: 512
    fast: 0.98641
    gpu: 0.22776
Size: 1024
    fast: 9.95456
    gpu: 1.04577
```

# Task 3.5: Training #
## SMALL (HIDDEN SIZE = 100) ##
### SIMPLE ###

### CPU - 0.156s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05

Epoch  10  loss  3.9169206554184868 correct 47
Epoch  20  loss  3.252112881612905 correct 48
Epoch  30  loss  2.3816200880570686 correct 49
Epoch  40  loss  0.8274228787967551 correct 49
Epoch  50  loss  1.7299644211852439 correct 49
Epoch  60  loss  2.057364066308265 correct 49
Epoch  70  loss  1.5742379556286965 correct 49
Epoch  80  loss  2.122043015107504 correct 49
Epoch  90  loss  0.7501367119159636 correct 49
Epoch  100  loss  0.6500834318474862 correct 49
Epoch  110  loss  1.8987241547898541 correct 49
Epoch  120  loss  1.083607202380331 correct 49
Epoch  130  loss  0.4277688154946465 correct 49
Epoch  140  loss  1.2744291225453828 correct 49
Epoch  150  loss  0.8720921203826414 correct 49
Epoch  160  loss  1.4009570220482568 correct 49
Epoch  170  loss  0.16787852190250566 correct 49
Epoch  180  loss  0.603468024386143 correct 49
Epoch  190  loss  1.077997866814744 correct 49
Epoch  200  loss  0.8397060443296913 correct 49
Epoch  210  loss  0.2792411614106766 correct 49
Epoch  220  loss  0.9069212131864804 correct 49
Epoch  230  loss  1.192569859428332 correct 49
Epoch  240  loss  0.2875566071862128 correct 49
Epoch  250  loss  1.6652956051308414 correct 49
Epoch  260  loss  0.2322525439926837 correct 50
Epoch  270  loss  0.9320331142165367 correct 49
Epoch  280  loss  1.013846406786894 correct 50
Epoch  290  loss  0.7515830824997459 correct 49
Epoch  300  loss  0.14465043453444815 correct 50
Epoch  310  loss  0.1711882351827292 correct 50
Epoch  320  loss  0.3370460998530709 correct 50
Epoch  330  loss  1.257840133869173 correct 50
Epoch  340  loss  0.6052100102890985 correct 50
Epoch  350  loss  0.35196652703310494 correct 50
Epoch  360  loss  0.7448097791631288 correct 50
Epoch  370  loss  1.0757108385769645 correct 50
Epoch  380  loss  0.7879385862899382 correct 50
Epoch  390  loss  0.14814650247187747 correct 50
Epoch  400  loss  0.47406722067841556 correct 50
Epoch  410  loss  0.1459401216537256 correct 50
Epoch  420  loss  0.8501355271894083 correct 50
Epoch  430  loss  0.1052930678547705 correct 50
Epoch  440  loss  0.4420685009148345 correct 50
Epoch  450  loss  0.39603346482273855 correct 50
Epoch  460  loss  0.6316505252450997 correct 50
Epoch  470  loss  0.4290385165731778 correct 50
Epoch  480  loss  0.6027579584046091 correct 50
Epoch  490  loss  0.17853801324863935 correct 50
Epoch  500  loss  0.22381371061113892 correct 50

real	1m18.741s
user	1m24.877s
sys	0m17.394s
```
### GPU - 1.71s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05

Epoch  10  loss  2.5767773464477237 correct 46
Epoch  20  loss  0.7015459464140565 correct 49
Epoch  30  loss  0.47260457858416605 correct 50
Epoch  40  loss  0.22246758599902247 correct 50
Epoch  50  loss  0.5724364766704563 correct 50
Epoch  60  loss  0.19350237941494192 correct 50
Epoch  70  loss  0.3798572050353858 correct 50
Epoch  80  loss  0.4998897610959099 correct 50
Epoch  90  loss  0.0038665701724203017 correct 50
Epoch  100  loss  0.7075857936112043 correct 50
Epoch  110  loss  0.3366546769389982 correct 50
Epoch  120  loss  0.08766463747114121 correct 50
Epoch  130  loss  0.13983959060202297 correct 50
Epoch  140  loss  0.27317677581582717 correct 50
Epoch  150  loss  0.13071613753121822 correct 50
Epoch  160  loss  0.19706729011166707 correct 50
Epoch  170  loss  0.008416382831270396 correct 50
Epoch  180  loss  0.0546724771254238 correct 50
Epoch  190  loss  0.1704789921824939 correct 50
Epoch  200  loss  0.301889162131727 correct 50
Epoch  210  loss  0.34100664051931523 correct 50
Epoch  220  loss  0.2803377693401563 correct 50
Epoch  230  loss  0.1725922358108178 correct 50
Epoch  240  loss  0.11092442080609763 correct 50
Epoch  250  loss  0.31385695519154244 correct 50
Epoch  260  loss  0.020125565737153368 correct 50
Epoch  270  loss  0.002215314121555171 correct 50
Epoch  280  loss  0.0015206822144718272 correct 50
Epoch  290  loss  0.20183097925839708 correct 50
Epoch  300  loss  0.0962469929739049 correct 50
Epoch  310  loss  0.10704161683460936 correct 50
Epoch  320  loss  0.10523517636016379 correct 50
Epoch  330  loss  0.15505725129715459 correct 50
Epoch  340  loss  0.14446205074323923 correct 50
Epoch  350  loss  0.07315991598370755 correct 50
Epoch  360  loss  0.00913342494018658 correct 50
Epoch  370  loss  0.01693038832403789 correct 50
Epoch  380  loss  0.08274539885815169 correct 50
Epoch  390  loss  0.06602587377335749 correct 50
Epoch  400  loss  0.14355326625368833 correct 50
Epoch  410  loss  0.05408066845706448 correct 50
Epoch  420  loss  0.14432239632029062 correct 50
Epoch  430  loss  0.03951093905619616 correct 50
Epoch  440  loss  0.08691462537259735 correct 50
Epoch  450  loss  0.032650277473117176 correct 50
Epoch  460  loss  0.016733722562424867 correct 50
Epoch  470  loss  0.024682762418342243 correct 50
Epoch  480  loss  0.022919920935140906 correct 50
Epoch  490  loss  0.04876068583874245 correct 50
Epoch  500  loss  0.03979517310174569 correct 50

real	14m15.062s
user	14m4.043s
sys	0m5.673s
```

### SPLIT ###

### CPU - 0.158s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05

Epoch  10  loss  5.074406216319006 correct 41
Epoch  20  loss  5.090491015114409 correct 45
Epoch  30  loss  5.372594890761875 correct 47
Epoch  40  loss  3.2494671657233787 correct 48
Epoch  50  loss  2.212860908876067 correct 49
Epoch  60  loss  1.9079465448773079 correct 48
Epoch  70  loss  2.5685220279468655 correct 48
Epoch  80  loss  3.102985993999581 correct 49
Epoch  90  loss  3.096451758071523 correct 48
Epoch  100  loss  0.9199110952180992 correct 49
Epoch  110  loss  2.1104421467977357 correct 50
Epoch  120  loss  0.8942344268981368 correct 50
Epoch  130  loss  0.6425090472326772 correct 50
Epoch  140  loss  1.0679573666919528 correct 48
Epoch  150  loss  1.3254735382857417 correct 49
Epoch  160  loss  0.7395227568011108 correct 48
Epoch  170  loss  2.0156876907343273 correct 50
Epoch  180  loss  0.5412612366308442 correct 48
Epoch  190  loss  1.0516437458131205 correct 49
Epoch  200  loss  1.3270558416533784 correct 50
Epoch  210  loss  1.056242596843843 correct 49
Epoch  220  loss  1.1570526367734086 correct 50
Epoch  230  loss  0.684436734305032 correct 49
Epoch  240  loss  1.0103095806858273 correct 50
Epoch  250  loss  1.3845491754741759 correct 50
Epoch  260  loss  0.6727339229927171 correct 50
Epoch  270  loss  0.3568522351210521 correct 49
Epoch  280  loss  0.5493270933766036 correct 50
Epoch  290  loss  0.20117560795186337 correct 50
Epoch  300  loss  0.9704489521891527 correct 50
Epoch  310  loss  0.3407186665788998 correct 50
Epoch  320  loss  0.30067723514785116 correct 50
Epoch  330  loss  0.6427465606407478 correct 48
Epoch  340  loss  0.42966466876713155 correct 50
Epoch  350  loss  0.7546771080796351 correct 50
Epoch  360  loss  1.2959172622797142 correct 50
Epoch  370  loss  0.2596115274371609 correct 50
Epoch  380  loss  1.315188101891391 correct 49
Epoch  390  loss  0.45590458106460097 correct 50
Epoch  400  loss  0.44457649330315974 correct 50
Epoch  410  loss  1.111123228401292 correct 50
Epoch  420  loss  0.5558302507639168 correct 50
Epoch  430  loss  1.185329493928251 correct 50
Epoch  440  loss  0.2213644860106418 correct 50
Epoch  450  loss  0.022140752325281378 correct 50
Epoch  460  loss  0.6823516852380164 correct 50
Epoch  470  loss  0.35834240711582455 correct 50
Epoch  480  loss  0.5027776870312066 correct 50
Epoch  490  loss  0.6670755997528575 correct 50
Epoch  500  loss  0.12025337180243972 correct 50

real	1m19.310s
user	1m25.824s
sys	0m17.711s
```
### GPU - 1.706s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05

Epoch  10  loss  3.880274559372965 correct 33
Epoch  20  loss  4.372451412853639 correct 44
Epoch  30  loss  4.327426391852205 correct 43
Epoch  40  loss  3.250318888415282 correct 44
Epoch  50  loss  4.535043889790699 correct 49
Epoch  60  loss  1.7837837202295086 correct 46
Epoch  70  loss  1.9335988748732207 correct 49
Epoch  80  loss  1.6213263621493523 correct 44
Epoch  90  loss  1.5746685132871958 correct 50
Epoch  100  loss  1.4899036303331514 correct 48
Epoch  110  loss  0.6233050258313874 correct 48
Epoch  120  loss  1.566654586823459 correct 49
Epoch  130  loss  2.1023156088153847 correct 50
Epoch  140  loss  2.7631356420219744 correct 50
Epoch  150  loss  0.6891829748965211 correct 48
Epoch  160  loss  0.9397652749600499 correct 49
Epoch  170  loss  2.0119582568378993 correct 49
Epoch  180  loss  1.0769971365214024 correct 49
Epoch  190  loss  0.9316852486543594 correct 50
Epoch  200  loss  1.8859597465854385 correct 49
Epoch  210  loss  0.6087820073060287 correct 49
Epoch  220  loss  0.5816701068508311 correct 49
Epoch  230  loss  1.6190969209234838 correct 48
Epoch  240  loss  0.33341843211332 correct 50
Epoch  250  loss  0.3841251117393696 correct 50
Epoch  260  loss  0.8505408038925294 correct 50
Epoch  270  loss  0.5974761930832398 correct 50
Epoch  280  loss  0.40370551688437595 correct 50
Epoch  290  loss  0.9402081353533747 correct 50
Epoch  300  loss  0.11949668639744183 correct 50
Epoch  310  loss  0.4634534250992799 correct 50
Epoch  320  loss  0.625106968929646 correct 50
Epoch  330  loss  0.8758937241784956 correct 50
Epoch  340  loss  0.5566520362249375 correct 49
Epoch  350  loss  0.1953544722695583 correct 49
Epoch  360  loss  0.4519030395135071 correct 50
Epoch  370  loss  0.3667394502427536 correct 50
Epoch  380  loss  0.8345518586017926 correct 50
Epoch  390  loss  0.24269224110805815 correct 50
Epoch  400  loss  0.2553419658721652 correct 50
Epoch  410  loss  0.02501080079141625 correct 50
Epoch  420  loss  0.4122310123140075 correct 50
Epoch  430  loss  0.22924356515761068 correct 49
Epoch  440  loss  0.6954045361948832 correct 50
Epoch  450  loss  0.6354224727564259 correct 50
Epoch  460  loss  0.41253483633891846 correct 50
Epoch  470  loss  0.2938217176893807 correct 50
Epoch  480  loss  0.08838313725184302 correct 50
Epoch  490  loss  0.4572096607607968 correct 50
Epoch  500  loss  0.46025097906553547 correct 50

real	14m13.946s
user	14m3.231s
sys	0m5.546s
```
### XOR ###

### CPU - 0.2s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.02
Epoch  10  loss  5.3415155256897044 correct 40
Epoch  20  loss  5.388300171340199 correct 42
Epoch  30  loss  5.051360980712741 correct 43
Epoch  40  loss  5.373375529415704 correct 42
Epoch  50  loss  5.003878958790303 correct 43
Epoch  60  loss  5.377708563537126 correct 45
Epoch  70  loss  4.022819263401411 correct 45
Epoch  80  loss  4.961756859014789 correct 44
Epoch  90  loss  3.9998392531335334 correct 46
Epoch  100  loss  3.490136298283074 correct 46
Epoch  110  loss  3.7852366661247454 correct 46
Epoch  120  loss  2.818789576164379 correct 46
Epoch  130  loss  3.1968209751802177 correct 46
Epoch  140  loss  3.326075807531442 correct 46
Epoch  150  loss  2.8844300320087233 correct 45
Epoch  160  loss  3.303943742974599 correct 45
Epoch  170  loss  3.2697700761007162 correct 44
Epoch  180  loss  2.703849831205351 correct 47
Epoch  190  loss  2.7164685893409417 correct 47
Epoch  200  loss  1.711113867174129 correct 47
Epoch  210  loss  2.552687850258533 correct 46
Epoch  220  loss  2.3673592555692906 correct 47
Epoch  230  loss  2.8062596217423894 correct 47
Epoch  240  loss  1.1807195475341907 correct 47
Epoch  250  loss  2.278384549852687 correct 47
Epoch  260  loss  2.602514148868978 correct 48
Epoch  270  loss  1.1688017953955918 correct 47
Epoch  280  loss  2.5955107329271887 correct 48
Epoch  290  loss  1.7648621492752308 correct 48
Epoch  300  loss  1.907007720132005 correct 48
Epoch  310  loss  1.1412242263037586 correct 48
Epoch  320  loss  2.0584536889580063 correct 48
Epoch  330  loss  1.6135535915094963 correct 48
Epoch  340  loss  1.8940443417768291 correct 49
Epoch  350  loss  1.9087185027703364 correct 49
Epoch  360  loss  1.4007357564220322 correct 50
Epoch  370  loss  1.5328508658775841 correct 50
Epoch  380  loss  1.879961039548378 correct 49
Epoch  390  loss  1.324875003509399 correct 49
Epoch  400  loss  1.6557638778669905 correct 49
Epoch  410  loss  1.795052123629156 correct 50
Epoch  420  loss  1.3724908787593402 correct 50
Epoch  430  loss  1.1983943912151895 correct 50
Epoch  440  loss  2.4152828666330093 correct 49
Epoch  450  loss  1.384382720507876 correct 50
Epoch  460  loss  1.0662092055230337 correct 50
Epoch  470  loss  1.2595587491143683 correct 50
Epoch  480  loss  1.5631460415059324 correct 50
Epoch  490  loss  1.4449740430244786 correct 50
Epoch  500  loss  0.7989776654022804 correct 50

real	1m40.670s
user	1m44.218s
sys	0m21.942s
```
### GPU - 1.7s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05

Epoch  10  loss  3.039010454979224 correct 41
Epoch  20  loss  3.1668773810122537 correct 40
Epoch  30  loss  3.4752855159242237 correct 49
Epoch  40  loss  2.699394005103899 correct 48
Epoch  50  loss  3.9121602826009996 correct 47
Epoch  60  loss  2.262131214330223 correct 49
Epoch  70  loss  2.5154978623702684 correct 48
Epoch  80  loss  2.514309667170985 correct 47
Epoch  90  loss  1.6482478587885017 correct 47
Epoch  100  loss  1.79127795588666 correct 49
Epoch  110  loss  1.510114336245915 correct 49
Epoch  120  loss  1.6367799459481343 correct 48
Epoch  130  loss  0.7244907363235824 correct 48
Epoch  140  loss  1.6114513162156616 correct 50
Epoch  150  loss  0.34993878255356486 correct 50
Epoch  160  loss  1.8999550236576215 correct 50
Epoch  170  loss  1.6514525220840752 correct 50
Epoch  180  loss  0.8149999646175127 correct 50
Epoch  190  loss  1.6468242505284507 correct 49
Epoch  200  loss  1.0804162212368635 correct 50
Epoch  210  loss  0.5774828915680167 correct 50
Epoch  220  loss  0.3428631663559939 correct 50
Epoch  230  loss  0.7463771680055237 correct 50
Epoch  240  loss  0.2491414531851178 correct 49
Epoch  250  loss  1.393588365543084 correct 49
Epoch  260  loss  1.0694351503207296 correct 50
Epoch  270  loss  0.4128042111782453 correct 50
Epoch  280  loss  1.066881123837001 correct 50
Epoch  290  loss  0.16946614831286255 correct 50
Epoch  300  loss  0.5458688941734935 correct 50
Epoch  310  loss  1.1675791259193922 correct 50
Epoch  320  loss  0.3351661883296102 correct 50
Epoch  330  loss  0.04189327903211616 correct 49
Epoch  340  loss  0.9000329597401617 correct 50
Epoch  350  loss  0.11942156912113056 correct 50
Epoch  360  loss  0.7630290972252742 correct 50
Epoch  370  loss  0.1394630936947698 correct 50
Epoch  380  loss  0.12181976799879718 correct 50
Epoch  390  loss  0.5845647827567718 correct 50
Epoch  400  loss  0.3645717194188025 correct 50
Epoch  410  loss  0.38863044134860725 correct 50
Epoch  420  loss  0.7164905074846274 correct 50
Epoch  430  loss  0.26899146625536907 correct 50
Epoch  440  loss  0.379340716842665 correct 50
Epoch  450  loss  0.32487588447493027 correct 50
Epoch  460  loss  0.9035132581511738 correct 50
Epoch  470  loss  0.05997561737452509 correct 50
Epoch  480  loss  0.6103011972012719 correct 50
Epoch  490  loss  0.25036771172069494 correct 49
Epoch  500  loss  0.1147048913292364 correct 50

real	14m10.054s
user	13m59.535s
sys	0m5.556s
```

## LARGE (HIDDEN SIZE = 200) ## 
### SIMPLE ###

### CPU - 0.306s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET simple --RATE 0.05

Epoch  10  loss  1.4698008451664706 correct 48
Epoch  20  loss  1.1101756501744122 correct 48
Epoch  30  loss  0.9215065747060497 correct 49
Epoch  40  loss  0.06304892087263222 correct 49
Epoch  50  loss  0.5740865850430404 correct 49
Epoch  60  loss  0.1814016102631335 correct 49
Epoch  70  loss  0.05583685445974714 correct 48
Epoch  80  loss  0.9417169086014784 correct 50
Epoch  90  loss  0.19598707397441897 correct 49
Epoch  100  loss  0.11774376225731976 correct 50
Epoch  110  loss  1.1758095018782384 correct 50
Epoch  120  loss  1.0069285837567155 correct 48
Epoch  130  loss  0.5914924272936878 correct 49
Epoch  140  loss  0.40992783749988854 correct 50
Epoch  150  loss  0.259657635689634 correct 50
Epoch  160  loss  0.3890706123230756 correct 50
Epoch  170  loss  0.28152894836953296 correct 50
Epoch  180  loss  0.37364506111609413 correct 50
Epoch  190  loss  0.5882584267038187 correct 50
Epoch  200  loss  0.34748553564302087 correct 50
Epoch  210  loss  0.6201606401996659 correct 50
Epoch  220  loss  0.7941270511984507 correct 49
Epoch  230  loss  0.0350501257852508 correct 50
Epoch  240  loss  0.5285855430624553 correct 50
Epoch  250  loss  0.5783615485239711 correct 50
Epoch  260  loss  0.5201431515473001 correct 50
Epoch  270  loss  0.5773669065117657 correct 50
Epoch  280  loss  0.09774617300502678 correct 50
Epoch  290  loss  0.4852077031068917 correct 50
Epoch  300  loss  0.20465352882046872 correct 50
Epoch  310  loss  0.18365535988171433 correct 50
Epoch  320  loss  0.5709957052098434 correct 50
Epoch  330  loss  0.006222820457358558 correct 50
Epoch  340  loss  0.00022854325980108025 correct 50
Epoch  350  loss  0.0085238254244731 correct 50
Epoch  360  loss  0.15074281493002004 correct 50
Epoch  370  loss  0.2278581613502152 correct 50
Epoch  380  loss  0.018023240795448724 correct 50
Epoch  390  loss  0.00011502459204689064 correct 50
Epoch  400  loss  0.06255027805766628 correct 50
Epoch  410  loss  0.28241895494651426 correct 50
Epoch  420  loss  0.0023298869258876983 correct 50
Epoch  430  loss  0.5953404405715284 correct 50
Epoch  440  loss  0.27083971449489347 correct 50
Epoch  450  loss  0.0002880423925168915 correct 50
Epoch  460  loss  0.02086140010106191 correct 50
Epoch  470  loss  0.12638112323324047 correct 50
Epoch  480  loss  0.0035934054576979226 correct 50
Epoch  490  loss  0.2548877281232619 correct 50
Epoch  500  loss  0.005271449043601013 correct 50

real	2m33.415s
user	2m55.177s
sys	0m30.636s
```
### GPU - 1.796s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET simple --RATE 0.05

Epoch  10  loss  1.7711982124733439 correct 49
Epoch  20  loss  0.8652121354283568 correct 49
Epoch  30  loss  0.33594942347831025 correct 50
Epoch  40  loss  0.06307798561306777 correct 50
Epoch  50  loss  1.1703699179784113 correct 50
Epoch  60  loss  0.6518572325421226 correct 50
Epoch  70  loss  0.3772517169847106 correct 50
Epoch  80  loss  0.5760773043650653 correct 50
Epoch  90  loss  0.40301100446263555 correct 50
Epoch  100  loss  0.2909361123263726 correct 50
Epoch  110  loss  0.5431047143502193 correct 50
Epoch  120  loss  0.4531269104288357 correct 50
Epoch  130  loss  0.36408732023026713 correct 50
Epoch  140  loss  0.21967090886218774 correct 50
Epoch  150  loss  0.33798301336896 correct 50
Epoch  160  loss  0.548774355491468 correct 50
Epoch  170  loss  0.026445931970803152 correct 50
Epoch  180  loss  0.6527147554891103 correct 50
Epoch  190  loss  0.5188248701784128 correct 50
Epoch  200  loss  0.008961317763859415 correct 50
Epoch  210  loss  0.00239123166454547 correct 50
Epoch  220  loss  0.11205309062852524 correct 50
Epoch  230  loss  0.018596112564359564 correct 50
Epoch  240  loss  0.014729765514557187 correct 50
Epoch  250  loss  0.49951558231716475 correct 50
Epoch  260  loss  0.27616003624982927 correct 50
Epoch  270  loss  0.0855806315934503 correct 50
Epoch  280  loss  0.031497706811470225 correct 50
Epoch  290  loss  0.1746983328851057 correct 50
Epoch  300  loss  0.03647582014576728 correct 50
Epoch  310  loss  0.20913747080186137 correct 50
Epoch  320  loss  0.1218322899687462 correct 50
Epoch  330  loss  0.013011942074264048 correct 50
Epoch  340  loss  0.07935389513102362 correct 50
Epoch  350  loss  0.08641009331173288 correct 50
Epoch  360  loss  2.6515739631279823e-05 correct 50
Epoch  370  loss  0.15664154301464678 correct 50
Epoch  380  loss  0.04656275213638491 correct 50
Epoch  390  loss  0.3946423912610909 correct 50
Epoch  400  loss  0.0027839730212270456 correct 50
Epoch  410  loss  0.00020325202511894383 correct 50
Epoch  420  loss  0.0016345547512648215 correct 50
Epoch  430  loss  0.021711163204531155 correct 50
Epoch  440  loss  0.16231994276058712 correct 50
Epoch  450  loss  0.07285383153993157 correct 50
Epoch  460  loss  0.10470087388799126 correct 50
Epoch  470  loss  0.00035971959987840524 correct 50
Epoch  480  loss  0.007058021368384037 correct 50
Epoch  490  loss  0.11313649459516621 correct 50
Epoch  500  loss  0.0018921932409586657 correct 50

real	14m58.786s
user	14m47.623s
sys	0m5.954s
```