function Multiplexes = Generate_Multiplexes (T) %T:tensor
[n1,n2,n3]=size(T);
v=[1:n3];
%generate possible permutations by fixing the first view
P = perms(v);
j=1; ppermutations=[];
for i=1:length(P)
    if P(i,1)==1
        ppermutations(j,:)=P(i,:);
        j=j+1;
    end
end
[l1,l2]=size(ppermutations);
%generate a set of multiplexes from a tensor T
Multiplexes={};
for i=1:l1
    k=2;
    Multiplex=[];
    Multiplex(:,:,1)=T(:,:,1);
    for j=1:l2-1
    Multiplex(:,:,k)= corr(T(:,:,ppermutations(i,j)),T(:,:,ppermutations(i,j+1)));
    Multiplex(:,:,k+1)= T(:,:,ppermutations(i,j+1));
    k=k+2;
    end
Multiplexes{i}=Multiplex;
end
end

