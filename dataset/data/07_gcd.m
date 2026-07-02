a = 48;
b = 18;
while b ~= 0
    tmp = b;
    b = mod(a, b);
    a = tmp;
end
fprintf('GCD is %d\n', a)
