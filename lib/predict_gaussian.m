function [labels_pred, scores_pred] = predict_gaussian(model,X)
%predict_gaussian Given a Gaussian CLassification model, it predicts the
%labels (and scores) of new samples given by X.

p_pos=mvnpdf(X,model.mean_pos,model.cov_pos);
p_neg=mvnpdf(X,model.mean_neg,model.cov_neg);

scores_pred = log(p_pos)-log(p_neg);
labels_pred = 2 .* (scores_pred > 0) - 1;
end

