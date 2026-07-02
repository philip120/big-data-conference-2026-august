n = 10;
fib = zeros(1, n);
fib(1) = 1;
fib(2) = 1;
for i = 3:n
    fib(i) = fib(i-1) + fib(i-2);
end
disp('Fibonacci:')
disp(fib)
