disp('Celsius -> Fahrenheit')
for c = 0:20:100
    f = c * 9/5 + 32;
    fprintf('%3d C = %6.1f F\n', c, f)
end
