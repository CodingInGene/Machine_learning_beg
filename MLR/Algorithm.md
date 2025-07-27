# Algorithm

1. Read data from csv -> data
2. Remove rows where any column data is missing
3. From the data seperate test columns to each array (Store each columns values to individual array). Here total_rooms, median_income and housing_median_age cols are used as X, median_house_value is used as Y
4. In X add all columns and 1 - [[1,x1,x2],[1,x1,x2]]. In Y same but without 1 - [y1,y2,y3]
5. Seperate test and training set

	**Model**
6. Get the co-efficients for the training set. b0,b1,b2,.....bn\
	**Formula, B = Inverse((transpose(X) * X)) * transpose(X) * Y.**\
	**B=((X<sup>T</sup> * X)<sup>-1</sup>) * X<sup>T</sup> * Y**

	**Prediction**
7. Based on the coefficients get the predictions,\
	**Formula, Ypredict = b0+b1x1+b2x2+....+bnxn**

8. On predict function send each row of test set and B(co-eff matrix).
9. Run a loop to add n indep. vars depending on the length of B->
		n=len(b)	#How many independent vars
    		res=b[0]
    		for i in range(1,n):	#Make equation for n indep. vars
        		res=(b[i]*x[i])+res
        		
10. Return each rows result from function and append results on main.

	**R2 score**
11. SSR = for i=1 to n, (ypredicted[i] - meanY)^2		# Sum of these
12. SST = for i=1 to n, (yactual[i] - meanY)^2		# Sum of these
11. r2 score = Sum of squares in regression(SSR) / Sum of squared total(SST)
12. Efficiency(%) = r2*100
