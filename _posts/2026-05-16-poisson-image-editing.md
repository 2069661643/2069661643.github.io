---
layout: blog_post
title: 'Poisson Image Editing: Seamless Cloning and Mixing Gradients'
date: 2026-05-16
tags:
  - Poisson
  - Image Editing
  - Computer Graphics
  - Seamless Cloning
  - Eigen
---

Poisson Image Editing is a gradient-domain technique that seamlessly blends a source image region into a target image. Unlike naive copy-and-paste, it preserves the source region's texture details while enforcing smooth boundary transitions. This post walks through the underlying mathematics, a sparse solver implementation, and experimental results.

## The Poisson Model

Given a source image $g$ and a target image $f^*$ defined on a domain $S \supset \Omega$, the goal is to construct a new image $f$ inside the region $\Omega$ such that:

- **Boundary condition**: $f|_{\partial\Omega} = f^*|_{\partial\Omega}$ — the pixel values at the boundary of $\Omega$ match the target image.
- **Gradient preservation**: $\nabla f \approx \nabla g$ — the internal gradient structure closely resembles that of the source.

This is formulated as a Poisson equation:

$$
\Delta f = \operatorname{div}(\nabla g)
$$

Discretizing yields a sparse linear system $Ax = b$, where $A$ is a Laplacian matrix (at most 5 non-zero entries per row), $x$ is the vector of unknown pixel values inside $\Omega$, and $b$ encodes the source gradients modified by boundary constraints.

### Discrete Form

For an interior pixel at coordinates $(x, y)$:

$$
\begin{aligned}
4f(x,y) - f(x-1,y) - f(x+1,y) - f(x,y-1) - f(x,y+1) =\\
4g(x,y) - g(x-1,y) - g(x+1,y) - g(x,y-1) - g(x,y+1)
\end{aligned}
$$

When a neighbour lies on $\partial\Omega$, the corresponding $f$ term is replaced with the known target image value $f^*$, moving it to the right-hand side.

## Mixing Gradients

A straightforward extension of Seamless Cloning. Instead of always using $\nabla g$, the mixing variant selects the stronger gradient at each pixel:

$$
\nabla f(p) = \begin{cases}
\nabla g(p) & \text{if } \|\nabla g(p)\| > \|\nabla f^*(p)\| \\
\nabla f^*(p) & \text{otherwise}
\end{cases}
$$

This preserves the source texture where it is prominent, while letting the target background show through in flat regions — producing more natural results.

## Implementation

### Sparse Matrix Construction with Eigen

The linear system is built using Eigen's triplet insertion:

```cpp
std::vector<Eigen::Triplet<double>> triplets;
for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
        if (!in_region(x, y)) continue;
        int idx = pixel_index(x, y);
        triplets.push_back({idx, idx, 4.0});
        for (auto [nx, ny] : neighbours(x, y)) {
            if (in_region(nx, ny))
                triplets.push_back({idx, pixel_index(nx, ny), -1.0});
            else
                rhs[idx] += target(ny, nx);  // boundary contribution
        }
    }
}
```

Each pixel contributes one equation with up to 5 non-zero coefficients. The RHS accumulates both source-gradients and boundary conditions.

### Solver Comparison

Multiple sparse solvers were evaluated across varying region sizes:

| Scale | SparseLU Compute | SparseQR Compute | BiCGSTAB Compute | SparseLU Solve | SparseQR Solve | BiCGSTAB Solve |
|------:|------:|-------:|-----:|-----:|------:|------:|
| 256   | 567   | 1,277  | 1    | 16.3 | 51.0  | 119.8 |
| 992   | 2,148 | 52,877 | 5    | 68.8 | 677.3 | 1,255.3 |
| 4,554 | 12,725| 3,388,841| 24| 370.8| 10,350.0| 12,060.3|
| 9,702 | 21,982| 29,438,131| 34| 814.5| 38,645.5| 39,686.0|

*Times in microseconds (μs). Compute = factorisation; Solve = average across 4 colour channels.*

SparseLU offers the best balance for interactive use: moderate factorisation cost with fast solves. BiCGSTAB has trivial factorisation but slower solve — suitable when the matrix changes every frame. SparseQR does not scale well for this problem.

### Real-Time Performance via Pre-factorisation

When the user drags the source region across the target image, the region shape stays constant — only the boundary values and source pixels change. This means:

- Matrix $A$ is **unchanged** — factorise once with `Eigen::SparseLU::compute(A)`
- For each frame, only update $b$ and call `Eigen::SparseLU::solve(b)`

This reuse cuts per-frame cost by orders of magnitude, enabling smooth real-time interaction:

![Real-time Poisson cloning]({{ '/assets/images/blog1-poisson-image-editing/realtime.gif' | relative_url }})

### Handling Complex Regions

Beyond rectangles, the system supports polygon, freehand, and ellipse regions. Interior pixel enumeration uses a **scanline algorithm**:

1. Treat the region boundary as a set of line segments.
2. For each image row, compute intersections with all segments.
3. Sort intersections and fill pixels in alternating (even-odd) spans.

This generalises the fill to arbitrary simple polygons with $\mathcal{O}(n \log n)$ per row.

## Results

### Seamless Cloning vs. Mixing Gradients

![Seamless Cloning comparison]({{ '/assets/images/blog1-poisson-image-editing/original_tar.png' | relative_url }})
![Source]({{ '/assets/images/blog1-poisson-image-editing/src_rect.png' | relative_url }})
![Seamless Cloning]({{ '/assets/images/blog1-poisson-image-editing/seamless_rect.png' | relative_url }})
![Mixing Gradients]({{ '/assets/images/blog1-poisson-image-editing/mixing_rect.png' | relative_url }})

Seamless Cloning preserves the source texture but can look "pasted" in uniform areas. Mixing Gradients adapts better by leveraging target gradients where the source offers little variation.

![My Seamless Cloning]({{ '/assets/images/blog1-poisson-image-editing/myseamless.jpg' | relative_url }})
![My Mixing Gradients]({{ '/assets/images/blog1-poisson-image-editing/mymixing.jpg' | relative_url }})

### Irregular Regions

The scanline algorithm handles non-rectangular masks:

![Ellipse]({{ '/assets/images/blog1-poisson-image-editing/ellipse.png' | relative_url }})
![Polygon]({{ '/assets/images/blog1-poisson-image-editing/polygon.png' | relative_url }})
![Freehand]({{ '/assets/images/blog1-poisson-image-editing/freehand.png' | relative_url }})

Poisson blending on irregular regions:

![Ellipse Seamless]({{ '/assets/images/blog1-poisson-image-editing/ellipse_seamless.png' | relative_url }})
![Polygon Seamless]({{ '/assets/images/blog1-poisson-image-editing/polygon_seamless.png' | relative_url }})
![Freehand Seamless]({{ '/assets/images/blog1-poisson-image-editing/freehand_seamless.png' | relative_url }})

## Summary

Poisson Image Editing turns a simple copy-paste into a physically plausible blend by solving a sparse Laplacian system. Key takeaways:

- **Seamless Cloning** preserves source gradients under Dirichlet boundary conditions.
- **Mixing Gradients** improves visual quality by selecting dominant gradients at each pixel.
- **SparseLU** with pre-factorisation is the sweet spot for interactive performance.
- **Scanline-based** region filling enables arbitrary polygon, freehand, and elliptical masks.

The implementation is built on a C++ framework using Eigen for sparse linear algebra. Solver choice and matrix reuse are critical for real-time editing.
