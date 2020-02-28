
% main program for brute force search of candidate panels using Dataset 1

disp(datetime('now'));

input_data_table = readtable('Dataset_1.csv');    

input_feature_list = {'A1M','A2M','ApoA2','ApoE','BNP','BTC','CD5L','Eot3','IGM','IL3','MCSF1','PAPPA','PLGF','PYY','RAGE','SGOT'}; % list of CFS selected markers from Dataset 1
header_class_labels = 'column_with_class_labels';         % specifies the column with class labels in Dataset 1

y_true = input_data_table.header_class_labels;
numeric_class_labels = unique(y_true);

neg_pos_class_labels = [1, 2];              % numeric class labels in the order [negative_class, positive_class]
cv_folds = 10;                              % number of cross-validation folds
cv_iterations = 10;                         % number of cross-validation iterations

panel_categs = gen_lists_of_panel_categs(input_feature_list,1,length(input_feature_list));   % brute force lists of potential panels grouped into different sizes (that we called categories(categs))

% loop over different categories (sizes) of panels
for panel_categ_ind = 1:length(panel_categs)
    
    fprintf('Biomarker category being processed is: %d\n', panel_categ_ind);
    panel_categ = panel_categs{1, panel_categ_ind};
    num_of_panels = length(panel_categ(:,1));

    % loop over different types of SVM kernels
    for potential_svm_kernels =  {'svmLinear', 'svmPolyQuad', 'svmPolyCubic', 'svmRBF'}   
       
        fprintf('Biomarker model being processed is: %s\n', char(potential_svm_kernels));
        performance_report_table = create_table(num_of_panels);      % for collecting the classification performanc of each panel
        row_to_write_to = 1;
        
        % loop over different panels
        for panel_ind = 1:num_of_panels
            fprintf('Panel_categ being processed is: %d\n', panel_ind);
            panel = {panel_categ{panel_ind,:}};  
            X = input_data_table(:, panel);
            accum_accuracy = 0;
            accum_sensitivity = 0;
            accum_specificity = 0;
            accum_auc = 0;
            
            % repeat kfold cross-validation, and randomise data used in D and Dt for each cross-validation 
            for iter = 1:cv_iterations
                rng(iter);              
                trained_model = train_svm(X,y_true,numeric_class_labels,char(potential_svm_kernels)); % for k-fold xval
                [y_pred,validation_scores,accuracy] = crossvalidate_model(trained_model, cv_folds);
                [sensitivity,specificity] = comp_sens_spec(y_true,y_pred,neg_pos_class_labels);
                [FPR,TPR,threshold,auc] = comp_auc(y_true,validation_scores,neg_pos_class_labels(2));
                %plot(FPR,TPR);
                accum_accuracy = accum_accuracy + accuracy;
                accum_sensitivity = accum_sensitivity + sensitivity;
                accum_specificity = accum_specificity + specificity;
                accum_auc = accum_auc + auc;
            end
            
            avg_accuracy = accum_accuracy/cv_iterations;
            avg_sensitivity = accum_sensitivity/cv_iterations;
            avg_specificity = accum_specificity/cv_iterations;
            avg_auc = accum_auc/cv_iterations;
            
            performance_report_table = write_perf_to_table(performance_report_table, panel,row_to_write_to,avg_accuracy, avg_sensitivity, avg_specificity, avg_auc); 

            row_to_write_to = row_to_write_to + 1; 
        end 
        
        write_table_to_csv(performance_report_table, char(potential_svm_kernels), panel_categ_ind);
        fprintf('Model that finished processing is: %s\n', char(potential_svm_kernels));
        disp(datetime('now'));
        
    end
end

