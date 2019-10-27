% Return the stability matrix, takes the features file in entry

function stab_mat = get_stability(data,final_tab)

    warning('off', 'stats:obsolete:ReplaceThisWithMethodOfObjectReturnedBy');
    warning('off', 'stats:obsolete:ReplaceThisWith');
    warning('off', 'stats:svmclassify:NoTrainingFigure');
    warning('off', 'stats:svmtrain:OnlyPlot2D');
    warning('off', 'MATLAB:singularMatrix');
    warning('off', 'MATLAB:illConditionedMatrix');
    warning('off', 'MATLAB:nearlySingularMatrix');
    
    final_tab = zeros(595,7);
    stab_mat = zeros(7,7);
    
    %for i=1:7
    %   final_tab(:,i) = get_ranking(data,i);
    %end
    final_tab = final_tab(1:200,:);
    for i=1:7        
         for j=1:7
            if i ~= j 
                stab_mat(i,j) = kuncheva_stability(horzcat(final_tab(:,i),final_tab(:,j)),595);
            end
         end
    end
      stab_mat = abs(stab_mat);
      stab_mat = normalise(stab_mat);
   % uncomment to plot stability matrix
   % figure
   % imagesc(stab_mat, [0 100])
   % colormap(brewermap([],'*Spectral')) 
   % title('stab mat')
   % colorbar
        
end

 function norm_mat = normalise(matrix)
    norm_mat = zeros(size(matrix,1),size(matrix,2));
    for i=1:size(matrix,1)
        for j=1:size(matrix,2)
            norm_mat(i,j) = 100*(matrix(i,j) - min(matrix(:)) ) / ( max(matrix(:)) - min(matrix(:)) );
        end
    end
 end
