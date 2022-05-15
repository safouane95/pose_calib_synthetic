gen_bin(1,2)


function gen_bin(s, e) % generate values in range by binary subdivision
    t = (s + e) / 2;
    lst = [s t; t e];

    for lst
        s, e = lst.pop(0);
        t = (s + e) / 2;
        lst.append(s, t);
        lst.append(t, e);
        yield t
    end
end