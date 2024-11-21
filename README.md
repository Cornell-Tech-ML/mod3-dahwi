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

### CPU - 0.134s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05

Epoch  10  loss  3.3252641224414434 correct 45 time per epoch 1.3690220355987548
Epoch  20  loss  2.5806959856576968 correct 45 time per epoch 0.7344612121582031
Epoch  30  loss  3.289439576113198 correct 44 time per epoch 0.5230983893076578
Epoch  40  loss  1.9167274886972088 correct 45 time per epoch 0.4179764032363892
Epoch  50  loss  1.1285786023158755 correct 46 time per epoch 0.3544344043731689
Epoch  60  loss  2.4286229792991794 correct 48 time per epoch 0.3121023694674174
Epoch  70  loss  2.630555814545342 correct 49 time per epoch 0.2870031833648682
Epoch  80  loss  1.8579657828443932 correct 50 time per epoch 0.27370076477527616
Epoch  90  loss  1.3871991842318718 correct 50 time per epoch 0.2544213745329115
Epoch  100  loss  1.1890027218080825 correct 50 time per epoch 0.2389927339553833
Epoch  110  loss  1.7364227744063019 correct 50 time per epoch 0.22647731520912864
Epoch  120  loss  1.372499694017122 correct 50 time per epoch 0.2159132917722066
Epoch  130  loss  1.5616803386326552 correct 50 time per epoch 0.20692038902869592
Epoch  140  loss  1.326570300160589 correct 50 time per epoch 0.19937769685472762
Epoch  150  loss  0.3963436484326189 correct 50 time per epoch 0.19276263554890952
Epoch  160  loss  0.7426700269113192 correct 50 time per epoch 0.18687539547681808
Epoch  170  loss  0.9038230805769649 correct 50 time per epoch 0.18176389862509335
Epoch  180  loss  1.4120681271297453 correct 50 time per epoch 0.17912695937686496
Epoch  190  loss  1.399171649843317 correct 50 time per epoch 0.17855089338202226
Epoch  200  loss  1.2648308043845822 correct 50 time per epoch 0.17558835744857787
Epoch  210  loss  0.7046432836157469 correct 50 time per epoch 0.17209228901636034
Epoch  220  loss  0.8539694919913513 correct 50 time per epoch 0.16874855865131724
Epoch  230  loss  0.9273219222615717 correct 50 time per epoch 0.1657171166461447
Epoch  240  loss  0.6380229723498929 correct 50 time per epoch 0.16294119755427042
Epoch  250  loss  0.2647268887448983 correct 50 time per epoch 0.1603956880569458
Epoch  260  loss  1.4015261834037696 correct 50 time per epoch 0.15807228271777812
Epoch  270  loss  0.7015690079818967 correct 50 time per epoch 0.15589502740789343
Epoch  280  loss  1.1537292939333 correct 50 time per epoch 0.15384659937449863
Epoch  290  loss  0.6009754952928954 correct 50 time per epoch 0.15205928786047573
Epoch  300  loss  0.9068812666555244 correct 50 time per epoch 0.15225114822387695
Epoch  310  loss  0.40573970782892765 correct 50 time per epoch 0.15218307279771373
Epoch  320  loss  0.7075965066236253 correct 50 time per epoch 0.1505004458129406
Epoch  330  loss  0.7995206607046169 correct 50 time per epoch 0.14892445188580136
Epoch  340  loss  1.2132203121068257 correct 50 time per epoch 0.14742707575068753
Epoch  350  loss  0.18589354334043479 correct 50 time per epoch 0.14601386138371059
Epoch  360  loss  0.6378597477874238 correct 50 time per epoch 0.1446773091952006
Epoch  370  loss  1.518217162711503 correct 50 time per epoch 0.1434528215511425
Epoch  380  loss  0.8348371024501731 correct 50 time per epoch 0.1423232448728461
Epoch  390  loss  1.0450024685135566 correct 50 time per epoch 0.1412079652150472
Epoch  400  loss  0.9687312993021467 correct 50 time per epoch 0.14018801093101502
Epoch  410  loss  0.5145231913517024 correct 50 time per epoch 0.1396902974058942
Epoch  420  loss  0.49997126531666536 correct 50 time per epoch 0.1411294897397359
Epoch  430  loss  0.9703606114352646 correct 50 time per epoch 0.14020764439605002
Epoch  440  loss  0.25351443936985235 correct 50 time per epoch 0.13931517655199224
Epoch  450  loss  0.8470246920462494 correct 50 time per epoch 0.138497355249193
Epoch  460  loss  0.0959097144580359 correct 50 time per epoch 0.1376667971196382
Epoch  470  loss  0.4321071281635691 correct 50 time per epoch 0.13689826346458273
Epoch  480  loss  0.18759607268889725 correct 50 time per epoch 0.1361595183610916
Epoch  490  loss  0.1806027739039822 correct 50 time per epoch 0.13551408660655118
Epoch  500  loss  0.1641437383053285 correct 50 time per epoch 0.1348092679977417

real	1m16.217s
user	1m22.584s
sys	0m17.106s
```
### GPU - 1.635s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05

Epoch  10  loss  2.6766845749289288 correct 49 time per epoch 1.925511121749878
Epoch  20  loss  1.3063792796390639 correct 50 time per epoch 1.7644101977348328
Epoch  30  loss  1.397139924897838 correct 50 time per epoch 1.7109199285507202
Epoch  40  loss  0.7260445238373606 correct 50 time per epoch 1.7081056237220764
Epoch  50  loss  0.7013557330297953 correct 50 time per epoch 1.685357871055603
Epoch  60  loss  0.16989423902920142 correct 50 time per epoch 1.690654162565867
Epoch  70  loss  0.566883413014451 correct 50 time per epoch 1.6963707617350987
Epoch  80  loss  0.5830964073618343 correct 50 time per epoch 1.6869630962610245
Epoch  90  loss  0.039137547301894414 correct 50 time per epoch 1.6803777456283568
Epoch  100  loss  0.24624676019168396 correct 50 time per epoch 1.6819026017189025
Epoch  110  loss  0.1414299389258757 correct 50 time per epoch 1.6750922549854625
Epoch  120  loss  0.045967222958538126 correct 50 time per epoch 1.6681975424289703
Epoch  130  loss  0.4250809036722911 correct 50 time per epoch 1.6683914734767034
Epoch  140  loss  0.08687518956302434 correct 50 time per epoch 1.6634655049868992
Epoch  150  loss  0.49632652391012266 correct 50 time per epoch 1.6585355440775553
Epoch  160  loss  0.06500288512760728 correct 50 time per epoch 1.6599420219659806
Epoch  170  loss  0.17001321956601262 correct 50 time per epoch 1.6571042159024407
Epoch  180  loss  0.15280708378762523 correct 50 time per epoch 1.6546613070699903
Epoch  190  loss  0.011128083526551792 correct 50 time per epoch 1.656103509350827
Epoch  200  loss  0.058657882892089085 correct 50 time per epoch 1.6539498102664947
Epoch  210  loss  0.26739883546913934 correct 50 time per epoch 1.650776473681132
Epoch  220  loss  0.12928449985568286 correct 50 time per epoch 1.6497785817493091
Epoch  230  loss  0.02260537537519361 correct 50 time per epoch 1.649017566183339
Epoch  240  loss  0.15877442108856243 correct 50 time per epoch 1.6460865239302318
Epoch  250  loss  0.16613300974114764 correct 50 time per epoch 1.6438010816574096
Epoch  260  loss  0.16222513829210536 correct 50 time per epoch 1.6447199683922988
Epoch  270  loss  0.26358713751336604 correct 50 time per epoch 1.645750543806288
Epoch  280  loss  0.1599900963363782 correct 50 time per epoch 1.644055802481515
Epoch  290  loss  0.07994062240619225 correct 50 time per epoch 1.6449059872791685
Epoch  300  loss  0.22833809222682586 correct 50 time per epoch 1.6429628872871398
Epoch  310  loss  0.028567492984114082 correct 50 time per epoch 1.6411303466366183
Epoch  320  loss  0.13122012726127247 correct 50 time per epoch 1.641872153431177
Epoch  330  loss  0.021585010099754415 correct 50 time per epoch 1.6403607953678478
Epoch  340  loss  0.01011042425397843 correct 50 time per epoch 1.6388014583026662
Epoch  350  loss  0.0003666482850360101 correct 50 time per epoch 1.6404170560836793
Epoch  360  loss  0.08291153558145384 correct 50 time per epoch 1.6399940954314338
Epoch  370  loss  0.07324848862710107 correct 50 time per epoch 1.6392282479518168
Epoch  380  loss  0.004627529939724397 correct 50 time per epoch 1.6392631675067701
Epoch  390  loss  0.0021332459775120494 correct 50 time per epoch 1.6391490728427203
Epoch  400  loss  0.11723010178257041 correct 50 time per epoch 1.6378543889522552
Epoch  410  loss  0.11468178592743576 correct 50 time per epoch 1.6363404698488189
Epoch  420  loss  0.02902382966955303 correct 50 time per epoch 1.6369529922803243
Epoch  430  loss  0.06851323663470599 correct 50 time per epoch 1.635834489312283
Epoch  440  loss  0.2248662013149479 correct 50 time per epoch 1.6346152300184422
Epoch  450  loss  0.0021593800782948266 correct 50 time per epoch 1.6356384542253282
Epoch  460  loss  0.02299532918473821 correct 50 time per epoch 1.635047453901042
Epoch  470  loss  0.06906726289477566 correct 50 time per epoch 1.636494517326355
Epoch  480  loss  0.0027151049547956716 correct 50 time per epoch 1.637769389152527
Epoch  490  loss  0.01839325451856551 correct 50 time per epoch 1.6368320937059364
Epoch  500  loss  0.02902948846708107 correct 50 time per epoch 1.6359553666114808

real	13m42.388s
user	13m32.854s
sys	0m4.957s
```

### SPLIT ###

### CPU - 0.142s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05

Epoch  10  loss  5.915829200994994 correct 42 time per epoch 1.4120561599731445
Epoch  20  loss  4.763276912146498 correct 41 time per epoch 0.7561843395233154
Epoch  30  loss  3.6977492564095242 correct 40 time per epoch 0.5382364670435588
Epoch  40  loss  4.145024414411303 correct 42 time per epoch 0.4299175262451172
Epoch  50  loss  4.2915423765709075 correct 45 time per epoch 0.384319806098938
Epoch  60  loss  3.8555599050430334 correct 49 time per epoch 0.3596012870470683
Epoch  70  loss  2.264056882046765 correct 47 time per epoch 0.32625637054443357
Epoch  80  loss  1.2297612553769253 correct 46 time per epoch 0.29811677932739256
Epoch  90  loss  1.6181886671968657 correct 48 time per epoch 0.27612871328989663
Epoch  100  loss  1.2334286990047783 correct 48 time per epoch 0.25875245094299315
Epoch  110  loss  1.7468097806842406 correct 50 time per epoch 0.24437914978374134
Epoch  120  loss  1.8230820664794487 correct 50 time per epoch 0.23248955408732097
Epoch  130  loss  0.5723085722004914 correct 48 time per epoch 0.22260457552396334
Epoch  140  loss  0.5751613312359105 correct 48 time per epoch 0.214230820110866
Epoch  150  loss  1.9041043216890585 correct 48 time per epoch 0.20675424734751383
Epoch  160  loss  0.8782064226754316 correct 50 time per epoch 0.2006910488009453
Epoch  170  loss  0.8652153781809924 correct 50 time per epoch 0.19918750173905317
Epoch  180  loss  0.7734809564373265 correct 49 time per epoch 0.19600740803612604
Epoch  190  loss  0.41635302759101317 correct 50 time per epoch 0.1910502320841739
Epoch  200  loss  0.9832467466026568 correct 49 time per epoch 0.18654623866081238
Epoch  210  loss  1.4170699379518035 correct 49 time per epoch 0.1825882219132923
Epoch  220  loss  1.0067376346327155 correct 49 time per epoch 0.17896117838946254
Epoch  230  loss  0.2455221094240599 correct 50 time per epoch 0.17562759337217912
Epoch  240  loss  1.0430924057810638 correct 50 time per epoch 0.17262251377105714
Epoch  250  loss  1.4038818487090219 correct 48 time per epoch 0.16994466495513916
Epoch  260  loss  0.5482712107199444 correct 50 time per epoch 0.16730082310163058
Epoch  270  loss  1.8400726596308563 correct 48 time per epoch 0.1648524487460101
Epoch  280  loss  0.24623078527762038 correct 49 time per epoch 0.16574565087045942
Epoch  290  loss  0.9714855413608059 correct 49 time per epoch 0.16436641216278075
Epoch  300  loss  0.3702201023603808 correct 49 time per epoch 0.16221951087315878
Epoch  310  loss  1.3754365936031872 correct 48 time per epoch 0.16023795066341276
Epoch  320  loss  0.9885843152478382 correct 50 time per epoch 0.15834343209862708
Epoch  330  loss  0.07290999740489448 correct 49 time per epoch 0.15662854584780606
Epoch  340  loss  0.4607481952322478 correct 50 time per epoch 0.15504872728796565
Epoch  350  loss  0.37097889202122475 correct 49 time per epoch 0.1535502631323678
Epoch  360  loss  0.164410039209164 correct 48 time per epoch 0.15211493505371942
Epoch  370  loss  0.3578836322783326 correct 50 time per epoch 0.15077363735920674
Epoch  380  loss  1.428241911696384 correct 47 time per epoch 0.1495051998841135
Epoch  390  loss  0.8843326347638866 correct 50 time per epoch 0.15036253256675525
Epoch  400  loss  0.28949122441792424 correct 50 time per epoch 0.15001529335975647
Epoch  410  loss  0.0508554247246102 correct 50 time per epoch 0.14885558442371646
Epoch  420  loss  0.24653962145192773 correct 50 time per epoch 0.1477309085073925
Epoch  430  loss  0.12401204748981995 correct 50 time per epoch 0.14664666375448537
Epoch  440  loss  0.31426249722325217 correct 50 time per epoch 0.14564656832001427
Epoch  450  loss  0.2041034457502562 correct 50 time per epoch 0.14468934853871662
Epoch  460  loss  0.08292006955463528 correct 50 time per epoch 0.14371588644774064
Epoch  470  loss  0.16899232636266126 correct 50 time per epoch 0.1428567389224438
Epoch  480  loss  0.04204577595967058 correct 50 time per epoch 0.14196280042330425
Epoch  490  loss  0.0988249057201895 correct 50 time per epoch 0.14109360198585355
Epoch  500  loss  0.0436046339452254 correct 50 time per epoch 0.14124609851837158

real	1m19.647s
user	1m24.556s
sys	0m18.402s
```
### GPU - 1.641s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05

Epoch  10  loss  6.187565338930693 correct 32 time per epoch 1.8498652219772338
Epoch  20  loss  3.4917775507041067 correct 40 time per epoch 1.7624830842018127
Epoch  30  loss  4.000259885142268 correct 41 time per epoch 1.711126708984375
Epoch  40  loss  3.125687075489413 correct 43 time per epoch 1.6844241321086884
Epoch  50  loss  3.754543857763969 correct 44 time per epoch 1.6758012104034423
Epoch  60  loss  4.35463286382845 correct 44 time per epoch 1.6709111293156942
Epoch  70  loss  1.3629558826921166 correct 45 time per epoch 1.6637007202420915
Epoch  80  loss  2.1680257610903118 correct 46 time per epoch 1.6602706044912339
Epoch  90  loss  2.0756496544846117 correct 46 time per epoch 1.661466572019789
Epoch  100  loss  3.0557230015800187 correct 47 time per epoch 1.6576749515533447
Epoch  110  loss  2.4421328062275043 correct 47 time per epoch 1.6565683863379739
Epoch  120  loss  5.569461279997353 correct 44 time per epoch 1.656804633140564
Epoch  130  loss  1.3887161461936623 correct 49 time per epoch 1.6516029871427096
Epoch  140  loss  2.5067234028558483 correct 49 time per epoch 1.6483750190053668
Epoch  150  loss  1.7181591091817672 correct 48 time per epoch 1.6500586986541748
Epoch  160  loss  1.0970110624306442 correct 48 time per epoch 1.646484151482582
Epoch  170  loss  0.8011345862706722 correct 49 time per epoch 1.6494914882323322
Epoch  180  loss  1.0709705558079439 correct 49 time per epoch 1.653402520550622
Epoch  190  loss  0.4445286188978064 correct 50 time per epoch 1.6517078474948281
Epoch  200  loss  2.4379247074369514 correct 47 time per epoch 1.6501590669155122
Epoch  210  loss  0.993910813657555 correct 48 time per epoch 1.6530024664742606
Epoch  220  loss  0.7226758231522975 correct 50 time per epoch 1.6504959756677802
Epoch  230  loss  0.3943989655165594 correct 48 time per epoch 1.6480797031651373
Epoch  240  loss  1.5651860249380423 correct 49 time per epoch 1.6491436918576559
Epoch  250  loss  0.2520476389952175 correct 47 time per epoch 1.6476724634170532
Epoch  260  loss  1.8109205532776114 correct 49 time per epoch 1.6458689781335685
Epoch  270  loss  0.9575114144181239 correct 50 time per epoch 1.6474387892970332
Epoch  280  loss  0.5816309673538198 correct 50 time per epoch 1.646122509241104
Epoch  290  loss  0.7616782893593137 correct 50 time per epoch 1.6452564872544386
Epoch  300  loss  1.1793419688059616 correct 48 time per epoch 1.6473063031832378
Epoch  310  loss  0.6417619858054294 correct 50 time per epoch 1.6463177550223567
Epoch  320  loss  1.0856961549543118 correct 50 time per epoch 1.645568836480379
Epoch  330  loss  0.5809236877318477 correct 50 time per epoch 1.6461903738253045
Epoch  340  loss  1.3201464290470253 correct 48 time per epoch 1.6446926446521983
Epoch  350  loss  0.22254276909106835 correct 50 time per epoch 1.6432644115175519
Epoch  360  loss  0.5711286428928982 correct 50 time per epoch 1.6430768569310505
Epoch  370  loss  0.8831982767175394 correct 50 time per epoch 1.6455981950502139
Epoch  380  loss  0.9532890472532147 correct 50 time per epoch 1.6441571423881933
Epoch  390  loss  0.1367679814167919 correct 49 time per epoch 1.6431752015382817
Epoch  400  loss  0.7556793940042592 correct 50 time per epoch 1.6438135474920272
Epoch  410  loss  0.9862921593589227 correct 50 time per epoch 1.6423291043537418
Epoch  420  loss  0.6101913019932311 correct 49 time per epoch 1.6410180018061684
Epoch  430  loss  0.6637694551607024 correct 50 time per epoch 1.64187387810197
Epoch  440  loss  0.24201159044163334 correct 50 time per epoch 1.6406013272025368
Epoch  450  loss  0.3721666457345188 correct 50 time per epoch 1.6395833079020183
Epoch  460  loss  0.7948124956616026 correct 50 time per epoch 1.6408538896104563
Epoch  470  loss  0.20645760994079024 correct 50 time per epoch 1.6406430787228523
Epoch  480  loss  0.24490146053491793 correct 50 time per epoch 1.6400770594676335
Epoch  490  loss  0.25043442961607515 correct 50 time per epoch 1.641557375752196
Epoch  500  loss  0.8422023894071657 correct 50 time per epoch 1.6412432327270507

real	13m46.702s
user	13m35.435s
sys	0m5.221s
```
### XOR ###

### CPU - 0.134s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.02

Epoch  10  loss  4.946189795209684 correct 41 time per epoch 1.3632052898406983
Epoch  20  loss  4.898629007142748 correct 43 time per epoch 0.7313896417617798
Epoch  30  loss  2.345307587068438 correct 45 time per epoch 0.520572829246521
Epoch  40  loss  4.0584362806329155 correct 45 time per epoch 0.4152993679046631
Epoch  50  loss  2.3178070904169217 correct 45 time per epoch 0.3521147441864014
Epoch  60  loss  2.769705323399414 correct 45 time per epoch 0.3099141279856364
Epoch  70  loss  1.2507924791960283 correct 45 time per epoch 0.2900473356246948
Epoch  80  loss  3.018617664116256 correct 45 time per epoch 0.2734458863735199
Epoch  90  loss  2.539246891790353 correct 45 time per epoch 0.25432462957170276
Epoch  100  loss  2.111701857482028 correct 46 time per epoch 0.2388203477859497
Epoch  110  loss  4.092438184295196 correct 46 time per epoch 0.22616182457317005
Epoch  120  loss  1.0767467587115351 correct 45 time per epoch 0.21560441652933757
Epoch  130  loss  1.1383866895794366 correct 46 time per epoch 0.20662347536820633
Epoch  140  loss  2.6517990035251886 correct 46 time per epoch 0.19888394219534739
Epoch  150  loss  1.5687870933680577 correct 46 time per epoch 0.1922432533899943
Epoch  160  loss  3.347188714505426 correct 45 time per epoch 0.18653714507818223
Epoch  170  loss  1.0888200871390077 correct 46 time per epoch 0.181397905069239
Epoch  180  loss  2.317589650536745 correct 46 time per epoch 0.17945367760128444
Epoch  190  loss  0.8181591669850475 correct 48 time per epoch 0.1793003094823737
Epoch  200  loss  3.846174504676541 correct 46 time per epoch 0.17530211806297302
Epoch  210  loss  1.7026611243678405 correct 46 time per epoch 0.17166175615219842
Epoch  220  loss  3.109819115799169 correct 48 time per epoch 0.1683426163413308
Epoch  230  loss  0.7910191557975743 correct 46 time per epoch 0.16527698247329048
Epoch  240  loss  1.8042273060061704 correct 48 time per epoch 0.1624688575665156
Epoch  250  loss  2.0549996972482107 correct 46 time per epoch 0.15993255138397217
Epoch  260  loss  1.5792044759859674 correct 46 time per epoch 0.15748484684870792
Epoch  270  loss  2.629664250980168 correct 46 time per epoch 0.15525415031998246
Epoch  280  loss  1.0166830483283797 correct 46 time per epoch 0.15330167327608382
Epoch  290  loss  2.6507497308433243 correct 48 time per epoch 0.15159536887859476
Epoch  300  loss  2.021008785172818 correct 46 time per epoch 0.15253047545750936
Epoch  310  loss  2.47224187387544 correct 48 time per epoch 0.15186202064637214
Epoch  320  loss  0.9576883707278797 correct 48 time per epoch 0.15019330829381944
Epoch  330  loss  1.5788496592187053 correct 46 time per epoch 0.14865515159838127
Epoch  340  loss  0.3633353496885083 correct 46 time per epoch 0.1472070848240572
Epoch  350  loss  0.9386051663041886 correct 46 time per epoch 0.1457999883379255
Epoch  360  loss  2.227148906884801 correct 49 time per epoch 0.1444922427336375
Epoch  370  loss  1.3687350955112942 correct 49 time per epoch 0.14329717996958138
Epoch  380  loss  1.4236792099456257 correct 46 time per epoch 0.14213332565207232
Epoch  390  loss  0.5523058670453377 correct 46 time per epoch 0.14103054939172208
Epoch  400  loss  1.2556150243234285 correct 49 time per epoch 0.13998291969299317
Epoch  410  loss  0.8233519734984818 correct 50 time per epoch 0.1402192266975961
Epoch  420  loss  0.4301583697138572 correct 46 time per epoch 0.14072294689360118
Epoch  430  loss  0.2974086810835587 correct 49 time per epoch 0.13975298460139784
Epoch  440  loss  1.2497715065556665 correct 50 time per epoch 0.13884935216470198
Epoch  450  loss  0.46127116230387316 correct 48 time per epoch 0.13800369845496283
Epoch  460  loss  0.40844347166214756 correct 50 time per epoch 0.13719639415326326
Epoch  470  loss  2.203158087411347 correct 49 time per epoch 0.13638356543601826
Epoch  480  loss  0.6312528904686807 correct 50 time per epoch 0.13558828632036846
Epoch  490  loss  0.8561848286021556 correct 50 time per epoch 0.13483460533375644
Epoch  500  loss  0.8436442515418106 correct 50 time per epoch 0.13411894369125366

real	1m15.596s
user	1m22.044s
sys	0m17.292s
```
### GPU - 1.639s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05

Epoch  10  loss  6.019973591810827 correct 43 time per epoch 1.8427761554718018
Epoch  20  loss  4.817132174987373 correct 34 time per epoch 1.749220621585846
Epoch  30  loss  9.940002011251003 correct 32 time per epoch 1.7154337167739868
Epoch  40  loss  3.8106097751766685 correct 46 time per epoch 1.6900771975517273
Epoch  50  loss  3.9290367262747616 correct 46 time per epoch 1.6775538969039916
Epoch  60  loss  4.642235860240476 correct 46 time per epoch 1.6766210039456686
Epoch  70  loss  2.119702801278808 correct 43 time per epoch 1.667181042262486
Epoch  80  loss  4.124302223463902 correct 46 time per epoch 1.6569582611322402
Epoch  90  loss  3.92491356118081 correct 44 time per epoch 1.6604709837171767
Epoch  100  loss  3.164183818860984 correct 46 time per epoch 1.6561561012268067
Epoch  110  loss  3.8050259661920567 correct 48 time per epoch 1.6536266608671708
Epoch  120  loss  2.0275781660216103 correct 47 time per epoch 1.6581930716832478
Epoch  130  loss  1.8242892634335326 correct 48 time per epoch 1.6560086947221022
Epoch  140  loss  2.1109990401138665 correct 48 time per epoch 1.6543167897633144
Epoch  150  loss  2.3646975968133828 correct 49 time per epoch 1.6567656596501668
Epoch  160  loss  1.4076700090154546 correct 48 time per epoch 1.6535432517528534
Epoch  170  loss  1.3350858926789704 correct 48 time per epoch 1.6504907299490537
Epoch  180  loss  1.8138297322185832 correct 48 time per epoch 1.658301395840115
Epoch  190  loss  2.5485124278222457 correct 48 time per epoch 1.6558753440254612
Epoch  200  loss  1.1114171615146486 correct 49 time per epoch 1.6545438730716706
Epoch  210  loss  0.37398091242582376 correct 49 time per epoch 1.6570054565157208
Epoch  220  loss  1.2082857733927304 correct 49 time per epoch 1.655136560310017
Epoch  230  loss  1.9602472648274691 correct 47 time per epoch 1.6530865306439606
Epoch  240  loss  1.4119780647295226 correct 50 time per epoch 1.6544061481952668
Epoch  250  loss  1.9575353607728 correct 49 time per epoch 1.6521187133789061
Epoch  260  loss  0.9796830055858381 correct 48 time per epoch 1.6499027022948631
Epoch  270  loss  0.5900745447308255 correct 50 time per epoch 1.6492956514711734
Epoch  280  loss  0.436071540695639 correct 50 time per epoch 1.6494347529751914
Epoch  290  loss  1.47821001439081 correct 50 time per epoch 1.6482667380365832
Epoch  300  loss  1.023415007463196 correct 50 time per epoch 1.6483932701746622
Epoch  310  loss  2.0501466976839295 correct 47 time per epoch 1.6487628144602622
Epoch  320  loss  1.9078875184601085 correct 50 time per epoch 1.6477058671414853
Epoch  330  loss  1.3112564301208065 correct 45 time per epoch 1.6467790719234583
Epoch  340  loss  0.5040003185563751 correct 50 time per epoch 1.6470013555358438
Epoch  350  loss  0.682378217485721 correct 49 time per epoch 1.645589325087411
Epoch  360  loss  1.338919230057176 correct 50 time per epoch 1.644159420993593
Epoch  370  loss  0.4996765038340251 correct 50 time per epoch 1.6449310850452732
Epoch  380  loss  0.4198416315121155 correct 50 time per epoch 1.6434574673050328
Epoch  390  loss  0.1521710132715236 correct 49 time per epoch 1.6423689768864558
Epoch  400  loss  0.3398460899842297 correct 50 time per epoch 1.6455801165103912
Epoch  410  loss  0.8737466057356243 correct 50 time per epoch 1.6443004503482725
Epoch  420  loss  0.26331131484127307 correct 50 time per epoch 1.6431394775708517
Epoch  430  loss  0.7738144301974349 correct 50 time per epoch 1.6430075512375943
Epoch  440  loss  0.3115039217239636 correct 50 time per epoch 1.6427015841007233
Epoch  450  loss  0.58719822974484 correct 50 time per epoch 1.641454152531094
Epoch  460  loss  0.5270615093191183 correct 50 time per epoch 1.6405507958453633
Epoch  470  loss  0.19947590682399813 correct 50 time per epoch 1.6414305626077854
Epoch  480  loss  0.20103349511011798 correct 50 time per epoch 1.6402135436733565
Epoch  490  loss  0.7204835546809062 correct 49 time per epoch 1.6392619152458348
Epoch  500  loss  0.12791370950286654 correct 50 time per epoch 1.6399961905479432

real	13m44.765s
user	13m34.371s
sys	0m5.013s
```

## LARGE (HIDDEN SIZE = 200) ##
### SIMPLE ###

### CPU - 0.277s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET simple --RATE 0.05

Epoch  10  loss  1.5788742082900702 correct 49 time per epoch 1.5159921646118164
Epoch  20  loss  2.0699758134879525 correct 48 time per epoch 0.8741628050804138
Epoch  30  loss  0.5363884412299551 correct 50 time per epoch 0.7008594592412313
Epoch  40  loss  0.7437290237494927 correct 49 time per epoch 0.5828783392906189
Epoch  50  loss  0.11064082404086562 correct 49 time per epoch 0.5124168252944946
Epoch  60  loss  1.082983309148973 correct 50 time per epoch 0.4654840270678202
Epoch  70  loss  0.7429176901935304 correct 50 time per epoch 0.4324420043400356
Epoch  80  loss  1.3817492390938222 correct 49 time per epoch 0.42217702567577364
Epoch  90  loss  0.910156785231591 correct 50 time per epoch 0.40085493988460963
Epoch  100  loss  0.18281186070380745 correct 50 time per epoch 0.38479854345321657
Epoch  110  loss  0.22033775429015148 correct 49 time per epoch 0.3709507660432295
Epoch  120  loss  0.04023318807504438 correct 50 time per epoch 0.35953163504600527
Epoch  130  loss  0.843182551227798 correct 50 time per epoch 0.358339707668011
Epoch  140  loss  0.26798345858654726 correct 49 time per epoch 0.3492423960140773
Epoch  150  loss  0.35626591728741275 correct 50 time per epoch 0.3416257540384928
Epoch  160  loss  0.2562376196504293 correct 50 time per epoch 0.33458822667598725
Epoch  170  loss  0.4449606924060791 correct 50 time per epoch 0.32931440718033733
Epoch  180  loss  0.003914547864395543 correct 50 time per epoch 0.32938986751768323
Epoch  190  loss  0.3446370480117811 correct 50 time per epoch 0.32398101405093543
Epoch  200  loss  0.9486946008016607 correct 50 time per epoch 0.31905619382858275
Epoch  210  loss  0.5059520475960395 correct 50 time per epoch 0.31455932231176464
Epoch  220  loss  0.4342522794240007 correct 50 time per epoch 0.3114442164247686
Epoch  230  loss  0.9083459611896596 correct 49 time per epoch 0.3118163616760917
Epoch  240  loss  0.000341042468925147 correct 49 time per epoch 0.3082471589247386
Epoch  250  loss  0.914752827792211 correct 49 time per epoch 0.30493644523620606
Epoch  260  loss  0.04353892520708263 correct 49 time per epoch 0.3019312702692472
Epoch  270  loss  0.8469332992097064 correct 49 time per epoch 0.2998159929558083
Epoch  280  loss  0.02749063144228619 correct 50 time per epoch 0.3004874587059021
Epoch  290  loss  0.17603176764342293 correct 50 time per epoch 0.2980147830371199
Epoch  300  loss  0.01387611298146623 correct 50 time per epoch 0.29558489481608075
Epoch  310  loss  0.00019573915577574216 correct 49 time per epoch 0.29340184042530676
Epoch  320  loss  0.00019397666889856626 correct 50 time per epoch 0.29150003269314767
Epoch  330  loss  0.04175469876118022 correct 50 time per epoch 0.29275814330939093
Epoch  340  loss  0.0480472209991855 correct 50 time per epoch 0.29079357385635374
Epoch  350  loss  0.021434648522709335 correct 50 time per epoch 0.28891491685594833
Epoch  360  loss  0.0006119661906987073 correct 49 time per epoch 0.2870843099223243
Epoch  370  loss  0.7232065186173129 correct 49 time per epoch 0.2858011993202003
Epoch  380  loss  0.5719272135983782 correct 50 time per epoch 0.2869400789863185
Epoch  390  loss  0.03462825026640258 correct 50 time per epoch 0.28554757008185755
Epoch  400  loss  0.0002686649502880969 correct 50 time per epoch 0.2841139042377472
Epoch  410  loss  0.02968393354693164 correct 50 time per epoch 0.2827152513876194
Epoch  420  loss  0.00031008167145753825 correct 50 time per epoch 0.282053774311429
Epoch  430  loss  0.00010351544520856229 correct 50 time per epoch 0.2829647097476693
Epoch  440  loss  0.012913629281372059 correct 50 time per epoch 0.28171119364825165
Epoch  450  loss  0.42302848756870387 correct 50 time per epoch 0.2805149624082777
Epoch  460  loss  0.0030959875501630225 correct 50 time per epoch 0.27932357943576314
Epoch  470  loss  0.023389440303852135 correct 50 time per epoch 0.2789480173841436
Epoch  480  loss  0.5561465812442359 correct 50 time per epoch 0.27977952857812244
Epoch  490  loss  0.0014608101735244033 correct 50 time per epoch 0.2788074644244447
Epoch  500  loss  0.12024685895555623 correct 50 time per epoch 0.27786592149734496

real	2m28.155s
user	2m50.152s
sys	0m30.384s
```
### GPU - 1.749s/epoch ###
```
!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET simple --RATE 0.05

Epoch  10  loss  1.9147320983499774 correct 46 time per epoch 1.882439637184143
Epoch  20  loss  0.6048482000314498 correct 50 time per epoch 1.8230287551879882
Epoch  30  loss  0.6082074125188278 correct 49 time per epoch 1.7770440181096394
Epoch  40  loss  0.5053771555363288 correct 48 time per epoch 1.7537761330604553
Epoch  50  loss  2.293642573599895 correct 50 time per epoch 1.7546561574935913
Epoch  60  loss  0.49691175157731815 correct 48 time per epoch 1.7473623911539713
Epoch  70  loss  2.5802669004141925 correct 48 time per epoch 1.7581770692552838
Epoch  80  loss  1.2810890400267425 correct 50 time per epoch 1.754627960920334
Epoch  90  loss  0.7433905278053462 correct 50 time per epoch 1.7728722360399034
Epoch  100  loss  0.06890580987068928 correct 49 time per epoch 1.7717735028266908
Epoch  110  loss  0.15706205421139127 correct 50 time per epoch 1.769227147102356
Epoch  120  loss  0.9029704351079957 correct 50 time per epoch 1.773866601785024
Epoch  130  loss  0.1008883934910772 correct 50 time per epoch 1.7700198503640983
Epoch  140  loss  0.03663527360489812 correct 50 time per epoch 1.7711706757545471
Epoch  150  loss  0.35341987766165267 correct 50 time per epoch 1.7658804988861083
Epoch  160  loss  0.5784976565801186 correct 50 time per epoch 1.7617133110761642
Epoch  170  loss  0.5551941180369706 correct 48 time per epoch 1.7627410019145293
Epoch  180  loss  0.02709946024486995 correct 50 time per epoch 1.7592096196280584
Epoch  190  loss  0.03476181581807408 correct 50 time per epoch 1.7607358029014186
Epoch  200  loss  0.1319440685850058 correct 50 time per epoch 1.7589532518386841
Epoch  210  loss  0.27477177312150575 correct 50 time per epoch 1.7574516682397752
Epoch  220  loss  0.011543982543203308 correct 50 time per epoch 1.7600121476433495
Epoch  230  loss  0.40239423612168196 correct 50 time per epoch 1.7575608429701433
Epoch  240  loss  0.4284981534117068 correct 50 time per epoch 1.7584962983926138
Epoch  250  loss  0.21796603657895341 correct 50 time per epoch 1.756482671737671
Epoch  260  loss  0.4372271930967777 correct 50 time per epoch 1.755512072489812
Epoch  270  loss  0.3145572214605179 correct 50 time per epoch 1.7559280616265756
Epoch  280  loss  0.45716117610094575 correct 50 time per epoch 1.7544761087213243
Epoch  290  loss  0.17424061732153445 correct 50 time per epoch 1.7591711644468637
Epoch  300  loss  0.06803529466349416 correct 50 time per epoch 1.7575652440388998
Epoch  310  loss  0.1559758548515355 correct 50 time per epoch 1.7588985466188
Epoch  320  loss  0.5063157581514963 correct 50 time per epoch 1.7589798510074615
Epoch  330  loss  0.33610254342194884 correct 50 time per epoch 1.757486556515549
Epoch  340  loss  0.02060341117399131 correct 50 time per epoch 1.7586263383136076
Epoch  350  loss  0.4682444785052608 correct 50 time per epoch 1.7574097810472762
Epoch  360  loss  0.04271045892658754 correct 50 time per epoch 1.7583650767803192
Epoch  370  loss  0.13083168822112193 correct 50 time per epoch 1.7566021210438496
Epoch  380  loss  0.28495914601261707 correct 50 time per epoch 1.7545405105540628
Epoch  390  loss  0.007238353249204683 correct 50 time per epoch 1.7550614680999364
Epoch  400  loss  0.060003938929139536 correct 50 time per epoch 1.7533060562610627
Epoch  410  loss  0.07199553035138423 correct 50 time per epoch 1.753430029822559
Epoch  420  loss  0.2424067087538448 correct 50 time per epoch 1.7515707299822854
Epoch  430  loss  0.004467342103900418 correct 50 time per epoch 1.7498997538588767
Epoch  440  loss  0.11221583148383889 correct 50 time per epoch 1.7502537684007125
Epoch  450  loss  0.0019001618322807467 correct 50 time per epoch 1.749210328525967
Epoch  460  loss  0.015804236146143083 correct 50 time per epoch 1.7497424068658247
Epoch  470  loss  0.00483771505662763 correct 50 time per epoch 1.74940275993753
Epoch  480  loss  0.09901896693157985 correct 50 time per epoch 1.7484758391976356
Epoch  490  loss  0.2439938546893606 correct 50 time per epoch 1.7504353216716222
Epoch  500  loss  0.15905773011297647 correct 50 time per epoch 1.749100398540497

real	14m39.541s
user	14m28.656s
sys	0m5.791s
```