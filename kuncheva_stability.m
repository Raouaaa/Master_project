function S = kuncheva_stability(featidx,d)
%  kuncheva Stability index r



featidx = featidx(1:200,:); %We only take 100 first values
k = size(featidx,2);
q = size(featidx,1);
r = NaN(q,q);
% kuncheva index r
for n = 2:q-1
    for m = n+1:q-1
        r(n,m) = length(intersect(featidx(n,1),featidx(n,2))) + length(intersect(featidx(n,1),featidx(n + 1,2))) + length(intersect(featidx(n,1),featidx(n-1,2)));
    end
end
A = (r-(k^2/d))./(k-(k^2/d));
A(isnan(A)) = 0;
S = 2.*sum(A(:))./(q*(q-1));
