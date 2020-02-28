
% main program for evaluation of generalization performance of candidate panels on the most stable and best kernel using Dataset 2 as test set

disp(datetime('now'));

train_set_table = readtable('Dataset_1.csv');
test_set_table = readtable('Dataset_2.csv');

candidate_panels_table = readtable('panels_that_met_criteria_from_brute_force_crossvalidate_svm_with_iteration_with_svmPolyQuad_kernel.csv');
list_of_panels = table2cell(candidate_panels_table(:,{'biomarkers'}));
header_class_labels = 'column_with_class_labels';

y_train = train_set_table.header_class_labels;
y_test =  test_set_table.header_class_labels;

numeric_class_labels = unique(y_train);
neg_pos_class_labels = [1, 2];              % numeric class labels in the order [negative_class, positive_class]

X_train= train_set_table(:,panel);
X_test = test_set_table(:,panel);

svm_kernel = 'svmPolyQuad';
  
performance_report_table = create_table(length(list_of_panels));
row_to_write_to = 1;

for panel_ind = 1:length(list_of_panels)
    fprintf('Panel index being processed is: %d\n', panel_ind);
    disp(datetime('now'));
    panel = split(list_of_panels(panel_ind)); % converts the string in the cell to a list
    X_train= train_set_table(:,panel);
    X_test = test_set_table(:,panel);
    trained_model = train_svm(X_train,y_train,numeric_class_labels,svm_kernel);
    [y_pred,pred_scores,accuracy] = make_prediction(trained_model,X_test,y_test);
    [sensitivity,specificity] = comp_sens_spec(y_test,y_pred,neg_pos_class_labels);
    [FPR,TPR,threshold,auc] = comp_auc(y_test,pred_scores,neg_pos_class_labels(2));
    %plot(FPR, TPR);
    performance_report_table = write_perf_to_table(performance_report_table,panel,row_to_write_to,accuracy,sensitivity,specificity,auc);
    row_to_write_to = row_to_write_to + 1;
end

write_table_to_csv(performance_report_table,svm_kernel,length(list_of_panels));

disp(datetime('now'));