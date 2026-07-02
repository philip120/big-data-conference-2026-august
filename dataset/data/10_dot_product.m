u = [1 2 3 4];
v = [4 3 2 1];
dot_product = sum(u .* v);
norm_u = sqrt(sum(u .^ 2));
norm_v = sqrt(sum(v .^ 2));
fprintf('Dot product: %d\n', dot_product)
fprintf('Norm of u: %.4f\n', norm_u)
fprintf('Norm of v: %.4f\n', norm_v)
