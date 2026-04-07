function Res = Sen_Spec_Acc1(result)

for T = 1 : length(unique(result(:,1)))
    tp = 0;tn =0;fp=0;fn=0;
   for i = 1 : size(result,1)
      if isequal(result(i,1),T)
          if isequal(result(i,2),T)
             tp = tp + 1;
          end
      end
    
      if ~isequal(result(i,1),T)
         if ~isequal(result(i,2),T)
            tn = tn + 1;
         end
      end
    
     if isequal(result(i,1),T)
        if ~isequal(result(i,2),T)
            fn = fn + 1;
        end
     end
    
     if ~isequal(result(i,1),T)
        if isequal(result(i,2),T)
            fp = fp + 1;
        end
     end
   end
   C(T,:) = [tp tn fp fn];
end


TP = mean(C(:,1));
TN = mean(C(:,2));
FP = mean(C(:,3));
FN = mean(C(:,4));

Sen = TP / (TP + FN);
Spec = TN /(TN + FP);
Acc = (TP+TN) / (TP+TN+FP+FN);
Precision = TP / (TP+FP);
Recall = TP / (TP+FN);
FMeasure = 2*((Precision*Recall)/(Precision+Recall));
NPV=TN/(TN + FN);
FPR = FP / (FP + TN);
FNR = FN / (FN + TP);
MCC = ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)); 
Res = [TP TN FP FN Sen Spec Acc Precision Recall FMeasure NPV FPR FNR MCC];

