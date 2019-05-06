Lazy evaluation
---------------

An expression such as `x + y * sin(z)` does not hold the result. **Values are only computed upon access or when the expression is assigned to a container**. This
allows to operate symbolically on very large arrays and only compute the result for the indices of interest:

``` {.sourceCode .}
// Assume x and y are xarrays each containing 1 000 000 objects
auto f = cos(x) + sin(y);

double first_res = f(1200);
double second_res = f(2500);
// Only two values have been computed
```

That means if you use the same expression in two assign statements, the computation of the expression will be done twice. It might be convenient to store the result of the expression in a temporary variable:

``` {.sourceCode .}
xt::xarray<double> tmp = cos(x) + sin(y);
xt::xarray<double> res1 = tmp + 2 * x;
xt::xarray<double> res2 = tmp - 2 * x;
```

Forcing evaluation
------------------

If you have to force the evaluation of an xexpression then you can use ``xt::eval`` on an xexpression.
Evaluating will either return a *rvalue* to a newly allocated container in the case of an xexpression, or a reference to a container in case you are evaluating a ``xarray`` or ``xtensor``. Note that, in order to avoid copies, you should use a universal reference on the lefthand side (``auto&&``). For example:


``` {.sourceCode .}
xt::xarray<double> a = {1, 2, 3};
xt::xarray<double> b = {3, 2, 1};
auto calc = a + b; // unevaluated xexpression!
auto&& e = xt::eval(calc); // a rvalue container xarray!
auto&& a_ref = xt::eval(a); // a reference to the existing container
```

Broadcasting
------------

In an operation involving two arrays of different dimensions, the array
with the lesser dimensions is broadcast across the leading dimensions of
the other. For example, if `A` has shape `(2, 3)`, and `B` has shape
`(4, 2, 3)`, the result of a broadcast operation with `A` and `B` has
shape `(4, 2, 3)`.

``` {.sourceCode .}
   (2, 3) # A
(4, 2, 3) # B 
---------
(4, 2, 3) # Result
```

The same rule holds for scalars, which are handled as 0-D expressions.
If A is a scalar, the equation becomes:

``` {.sourceCode .}
       () # A
(4, 2, 3) # B 
---------
(4, 2, 3) # Result
```

If matched up dimensions of two input arrays are different, and one of
them has size `1`, it is broadcast to match the size of the other. Let's
say B has the shape `(4, 2, 1)` in the previous example, so the
broadcasting happens as follows:

``` {.sourceCode .}
   (2, 3) # A
(4, 2, 1) # B 
---------
(4, 2, 3) # Result
```

Layout types
==================

`xtensor` provides a `layout_type` enum that helps to specify the layout
used by multi-dimensional arrays:

-   `layout_type::row_major`, `layout_type::column_major` : fixes the strided index scheme.
-   `layout_type::dynamic`: `resize` and constructor overloads allow specifying a set of strides.

Examples:

``` {.sourceCode .}
std::vector<size_t> shape = { 3, 2, 4 };
std::vector<size_t> strides = { 8, 4, 1 };
xt::xarray<double, xt::layout_type::dynamic> a(shape, strides);
xt::xarray<double, xt::layout_type::dynamic> b(shape, xt::layout_type::row_major);

xt::xarray<double, xt::layout_type::row_major> c(shape);
// same:
xt::xarray<double> c(shape);
```

Containers Classes
======================================

Runtime vs Compile-time dimensionality:

-   `xarray` can be reshaped dynamically to any number of dimensions. It
    is the container that is the most similar to numpy arrays.
-   `xtensor` has a dimension set at compilation time, which enables
    many optimizations. For example, shapes and strides of `xtensor`
    instances are allocated on the stack instead of the heap.
-   `xtensor_fixed` has a shape fixed at compile time. This allows even
    more optimizations, such as allocating the storage for the container
    on the stack, as well as computing strides and backstrides at
    compile time, making the allocation of this container extremely
    cheap.

Example:

``` {.sourceCode .}
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"

xt::xarray<double> a({3, 2, 4});
xt::xtensor<double, 3> a({3, 2, 4});
xt::xtensor_fixed<double, xt::xshape<3, 2, 4>> a();
```

`xarray`, `xtensor` and `xtensor_fixed` containers are all `xexpression`. 
Additional interface they provide:

-   Each method exposed in `xexpression` interface has its non-const
    counterpart exposed by `xarray`, `xtensor` and `xtensor_fixed`.
-   `reshape()`: the new shape has to have same number of elements as the original container,
     keeps old elements.
-   `resize()`: resizes the container, doesn't preserve the container elements.
-   `strides()`: returns the strides of the container.

``` {.sourceCode .}
// Tip: The shape argument can have one of its value equal to -1, 
//      in this case the value is inferred from the number of elements 
//      in the container and the remaining values in the shape
xt::xarray<int> a = { 1, 2, 3, 4, 5, 6, 7, 8 };
a.reshape({-1, 4});
//a.shape() is {2, 4}
```

Aliasing and temporaries
========================

In the following example, temporary variable will be created to store `a + b + c` before assigning to b, because b can be required to resize:
``` {.sourceCode .}
b = a + c + b;
```

If the left-hand side is not involved in the expression being assigned, to prevent the usage of temporary variable use `xt::noalias()`:
``` {.sourceCode .}
xt::noalias(b) = a + c;
// Even if b has to be resized, a+c will be assigned directly to it
```

Scalar assignment
========================
In xtensor scalars are assumed to be 0-D expressions, so `a = 1.5` will lead to resizing `a` to 0 dimensions and assigning `1.5`.
The correct way to assign scalar is using `.fill()` method:
``` {.sourceCode .}
a = 1.5;     // 0-D xarray
a.fill(1.5); // N-D xarray, all values equal 1.5
```
For the reasoning behind such approach read [documentation](https://xtensor.readthedocs.io/en/latest/scalar.html).


Operators and Functions
========================
Operators
----------------------
- `+, -, *, /, %, !, ||, &&, <, <=, >, >=` are element-wise operators and apply the lazy broadcasting rules.
- `==, !=` return single `true` or `false`.

Functions
----------------------
From `namespace xt` :
- math: `pow`, `sin`, `ceil` ...
- casting: `cast<>`, performs `static_cast<>`;
- reducers: `sum`, `reduce`...
- accumulators: `cumsum`, `accumulate`...

Views
=====

`xt::view`
------------


Slices can be specified in the following ways:

-   selection in a dimension by specifying an index (unsigned integer)
-   `range(min, max)`, a slice representing the interval [min, max)
-   `range(min, max, step)`, a slice representing the stepped interval [min, max)
-   `all()`, a slice representing all the elements of a dimension
-   `newaxis()`, a slice representing an additional dimension of length one
-   `keep(i0, i1, i2, ...)` a slice selecting non-contiguous indices to keep on the underlying expression
-   `drop(i0, i1, i2, ...)` a slice selecting non-contiguous indices to drop on the underlying expression

``` {.}
xt::view(a, xt::range(1, 3), xt::all(), xt::range(1, 3));
xt::view(a, 1, xt::all(), xt::range(0, 4, 2));
xt::view(a, xt::all(), xt::all(), xt::newaxis(), xt::all());
xt::view(a, xt::drop(0), xt::all(), xt::keep(0, 3));
```

The range function supports the placeholder `_` syntax:

``` {.}
using namespace xt::placeholders;  // required for `_` to work
xt::view(a, xt::range(_, 2), xt::all(), xt::range(1, _));
```