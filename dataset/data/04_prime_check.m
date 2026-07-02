num = 37;
is_prime = true;
if num < 2
    is_prime = false;
end
for i = 2:floor(sqrt(num))
    if mod(num, i) == 0
        is_prime = false;
        break
    end
end
if is_prime
    fprintf('%d is prime\n', num)
else
    fprintf('%d is not prime\n', num)
end
