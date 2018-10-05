import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import pandas_datareader.data as wb
import pdb
import datetime as dt
import sys

class AT:

	def cumulative(self,returns):
		cum=(returns+1).cumprod(axis=0)
		return cum

	def Drawdown(self,cum):
		peak=np.maximum.accumulate(cum)
		ddx=cum/peak
		return ddx
		
	def MaxDrawdown(self,DD):
		MaxDD=DD.min(axis=0)-1
		if DD.ndim > 1:
			#MaxDDdate=np.array([DD.index[DD.iloc[:,x]==DD.iloc[:,x].min()] if DD.iloc[:,x].min() != 1 else 1 for x in np.arange(len(DD.columns))])
			MaxDDdate=DD.apply(lambda x: x.argmin(axis=0))
			res=pd.DataFrame([MaxDD.values,MaxDDdate.apply(lambda x: x.date())],columns=DD.columns).T
		else:
			#MaxDDdate=np.array([DD.index[DD==DD.min()] if DD.min() != 1 else 1])
			MaxDDdate=DD.argmin()
			res = pd.DataFrame([MaxDD,MaxDDdate.date()]).T
		
		return res

	def Recover(self,cum):
		
          def Rec(cum1):
		  #Test line
              #Drawdown profile for entire price series
              DD = Drawdown(cum1)-1
              #Max price and peak date (first occurance within group) (grouped by accumulating maximum)
              Allpeakper1 = cum1.loc[cum1.groupby(np.maximum.accumulate(cum1)).idxmax()]
              #Min price and trough date (grouped by accumulating maximum)
              Allpeakper2 = cum1.loc[cum1.groupby(np.maximum.accumulate(cum1)).idxmin()]     
              #Min price and trough date (removing dates which feature on min and max list)        
              Peakper = Allpeakper2[Allpeakper2.index != Allpeakper1.index]
              #first date for each group, indexed by accumulating maximum
              Allstartdates = cum1.groupby(np.maximum.accumulate(cum1)).apply(lambda x: x.first_valid_index())
              #first dates, only keeping dates wehere max price != min price
              DDstartdates =  Allstartdates[Allstartdates.index != Allpeakper2]
              #end date for each group, indexed by accumulating maximum
              Allenddates = cum1.groupby(np.maximum.accumulate(cum1)).apply(lambda x: x.last_valid_index())
              #end date, only keeping dates wehere max price != min price
              DDenddates =  Allenddates[Allenddates.index != Allpeakper2]
              #Drawdown value on trough dates
              DDper = DD[Peakper.index]
              #length of all groups
              AllDDperlen = cum1.groupby(np.maximum.accumulate(cum1)).count()
              #DD length calculated as group count minus one. Only keeping periods where max != min
              DDperlen = AllDDperlen[AllDDperlen.index != Allpeakper2]-1
              #Results table
              Allres = pd.DataFrame(
                                    [
                                    DDper.values,DDperlen.values,DDper.index.date,
                                    DDstartdates.dt.date,DDenddates.dt.date
                                    ],
                                    index=
                                    [
                                    "DD","DD Length","Trough Date",
                                    "Start Date","End Date"
                                    ]
                                    ).T
              return Allres
		
              if cum.ndim > 1:
                  Recs= pd.concat([Rec(cum[x]) for x in cum.columns],axis=1)
              else:
                  Recs=Rec(cum)
              return Recs
		
	def All(self,ret):
		Allcum=self.cumulative(ret)
		AllDD=self.Drawdown(Allcum)
		MaxAllDD=self.MaxDrawdown(AllDD)
		results = pd.concat([Allcum,AllDD],axis=1)
		
		return results,MaxAllDD

	def PeriodicDD(self,ret,period):
		
		def Insertzeros(periodret):
			if periodret.ndim >1:
				newret = pd.DataFrame(np.insert(periodret.values,0,np.zeros([1,len(periodret.columns)]),axis=0),index=np.insert(periodret.index,0,0))
			else:
				newret = pd.DataFrame(np.insert(periodret.values,0,0,axis=0),index=np.insert(periodret.index,0,0)) 
			return newret
		
		Periodic=ret.groupby(np.arange(len(ret))//period).apply(lambda x:All(Insertzeros(x))[1])
		
		if ret.ndim >1:
		
			PeriodicDrawdowns=Periodic[0].unstack(level=1)
			PeriodicDrawdownDates= Periodic[1].unstack(level=1)
			PeriodicDrawdowns.columns=ret.columns
			PeriodicDrawdownDates.columns=ret.columns
			Perdates= ret.index[period-1::period].to_series()
			Perdates.loc[ret.index[-1]]=ret.index[-1]
			PeriodicDrawdowns.index=Perdates
			PeriodicDrawdownDates.index=Perdates
			perDD=pd.concat([PeriodicDrawdowns,PeriodicDrawdownDates],axis=1)
			
		else:
		
			PeriodicDrawdowns=Periodic[0]
			PeriodicDrawdownDates= Periodic[1]
			Perdates= ret.index[period-1::period].to_series()
			Perdates.loc[ret.index[-1]]=ret.index[-1]
			PeriodicDrawdowns.index=Perdates
			PeriodicDrawdownDates.index=Perdates
			perDD = pd.DataFrame([PeriodicDrawdowns,PeriodicDrawdownDates],index=[ret.name + " MaxDD",ret.name+" MaxDDdate"]).T
				
		return perDD

	def AnnualDD(self,ret):
		AnnDD=ret.groupby(ret.index.year).apply(lambda x: All(x)[1])[0].unstack(level=1)
		AnnDDdates=ret.groupby(ret.index.year).apply(lambda x: All(x)[1])[1].unstack(level=1)
		
		if ret.ndim > 1:
			AnnDD.columns=ret.columns
			AnnDDdates.columns = ret.columns
			drawdowns = pd.concat([AnnDD,AnnDDdates],axis=1)
		else:
			drawdowns = pd.DataFrame([AnnDD,AnnDDdates])
		
		return drawdowns
		
	def AnnualRet(self,ret):
		ret1=ret+1
		years=np.unique(ret1.index.year)
		Ann = pd.concat([ret1[ret1.index.year==y].prod() for y in years],axis=1).T - 1
		Ann.index=years
		return Ann

	def Getcash(self,datesdf,DorM):
		cashfile=r"C:\Users\ato\Documents\Ainsley Python\DATA\Risk Free.xlsx"
		if DorM == "D":
			cashdata=pd.read_excel(cashfile,sheetname="Daily")
			cash = cashdata.rename(index=cashdata.iloc[:,0]).drop("Date",1)
			
		elif DorM == "M":
			cashdata=pd.read_excel(cashfile,sheetname="Monthly")
			cash = cashdata.rename(index=cashdata.iloc[:,0]).drop("Date",1)
		return cash.loc[datesdf]
		
	def RollingSharpe(self,ret, rf,window,days):
		rollannret=(ret+1).rolling(window).apply(lambda x:np.prod(x))**(days/window) -1 
		rollvol = ret.rolling(window).std()*np.sqrt(days)
		rollsharpe = ((rollannret.values.T-rf.loc[rollannret.index].values.T)/rollvol.T).T
		return rollsharpe
		
	def Stats(self,ret,per,annual):
		if ret.ndim > 1:
			vol=ret.std()*np.sqrt(annual)
			cum=self.cumulative(ret)
			Perret=cum[::per].pct_change()
			TR=cum.iloc[-1]**(annual/len(ret))-1
			MaxDD=self.Drawdown(cum).min(axis=0) -1
			Bestper = Perret.max(axis=0)
			Worstper = Perret.min(axis=0)
			PercentPos=Perret[Perret>0].count(axis=0)/Perret.count(axis=0)
		else:
			vol=ret.std()*np.sqrt(annual)
			cum=self.cumulative(ret)
			Perret=cum[::per].pct_change()
			TR=cum.iloc[-1]**(annual/len(ret))-1
			MaxDD=self.Drawdown(cum).min() -1
			Bestper = Perret.max()
			Worstper = Perret.min()
			PercentPos=Perret[Perret>0].count()/Perret.count()
		return pd.DataFrame([TR,vol,MaxDD,Bestper,Worstper,PercentPos],index=["AnnTR","AnnVol","MaxDD","Best Period","Worst Period","%PeriodsPositive"])
		
	def AAmix(self,ret):
		weights=pd.Series(np.zeros(len(ret.columns)),index=ret.columns)
		print(ret.columns)
		for cols in ret.columns:
			weights[cols]=float(input("Weight of "+cols+" in decimals:"))
			print("Total Allocation so far  " + str(weights.sum()))
		
		#ones=pd.DataFrame(np.ones_like(ret),index=ret.index,columns=ret.columns)
		return (ret*weights).sum(axis=1),weights
	
	def TailRet(self,rets):
		'''Creates a DF of returns, separating the high/low tail returns and bulk returns'''
		quant=float(input("Quantile to define tails (decimal):"))
		def tret(ret):
			Hightail=ret[ret>ret.quantile(1-quant)]
			Lowtail=ret[ret<ret.quantile(quant)]
			Alltail=ret[(ret>ret.quantile(1-quant))|(ret<ret.quantile(quant))]
			Bulk=ret[(ret<ret.quantile(1-quant))&(ret>ret.quantile(quant))]
			final= pd.concat([Hightail,Lowtail,Alltail,Bulk],axis=1)
			final.columns=[str(ret.name)+" Hightail",str(ret.name)+" Lowtail",str(ret.name)+" Alltail",str(ret.name)+" Bulk"]
			return final
		if rets.ndim >1:
			res=pd.concat([tret(rets[x]) for x in rets.columns],axis=1)
		else:
			res=tret(rets)
		return res
	
	def DrawdownDist(self,rets,rollper,**histbin):
		'''Creates a drawdown distribution with 10% bins to -100, unless kwarg entered'''
		manbin=histbin["histbin"]
		
		DDs=rets.rolling(rollper).apply(lambda x: self.Drawdown(self.cumulative(x)).min(axis=0)-1).dropna()
		
		if len(manbin) >0:
			DDdec=DDs.sort_values(ascending=False).value_counts(bins=manbin).apply(lambda x:x/len(DDs))
		else:
			DDdec=DDs.sort_values(ascending=False).value_counts(bins=np.arange(-1,.1,.1)).apply(lambda x:x/len(DDs))
		#groupby(pd.Series(np.linspace(1,11,num=len(DDs),dtype=int,endpoint=False))).last()
		return DDdec,DDs
	
	def RollVar(self,rets,per):
		'''Calculates rolling Value At Risk based on inputted quantile and periods'''
		quant=float(input("VaR quantile in decimals (0.05): "))

		#scaling VaR for confidence intervals: var(95%)=coeff(95)/coeff(99) * var(99%)...99%,97.5%,95% have 2.326,1.96,1.645 as coefficients of normal distribution
		#scaling VaR limit of different time periods: VaR (5d) = np.sqrt(5)/np.sqrt(20) * VaR (20d)
		rolls=rets.rolling(per).apply(lambda x: np.percentile(x,(quant*100))).dropna()
		return rolls
	
	def ProbSuccess(self,rets,DorM):
		'''From Returns Calculates the Probability of a positive outcome for different rolling periods. Need DorM for annualised'''
		if DorM=="D":
			annper=252
		else:
			annper=12
		years=[int(q) for q in input("List of rolling periods (no spaces) e.g. 12,60,120: ").split(",")]
		ret1=rets.apply(lambda x:x+1)
		rolls=pd.concat([ret1.rolling(pers).apply(lambda x: (x.prod()**(annper/pers))-1).dropna() for pers in years],axis=1)
		rolls.columns=np.concatenate(np.array([rets.columns+str(y)+" roll" for y in years]))
		probs = rolls[rolls>=0].count()/rolls.count()
		return rolls,probs
		
	def NormRet(self,rets):
		'''De-meaned returns divided by StdDev'''
		DWM=input("Daily or Weekly or Monthly (D/W/M)?: ")
		if DWM == "D":
			DorM =252
		elif DWM == "W":
			DorM=52
		else:
			DorM=12
		inputvol=float(input("Ann Vol to normalise to: "))/np.sqrt(DorM)
		inputret=(1+float(input("Ann Return to normalise to: ")))**(1/DorM)-1
		actvol=rets.std()
		actmeanret=rets.mean()
		normedcalc=(rets-actmeanret)/actvol
		normedrets=inputret + inputvol*normedcalc
		return normedrets

	def BootStrap(self,rets):
		'''Draws random returns and returns cumulative and drawdown. Preserves same random draws for each series in rets'''
		
		Periods=int(input("Number of Periods to compound returns over:"))
		Runs=int(input("Number of runs:"))
		#min=input("Minimum Value in runs (e.g. replace below 0 with 0:")
		#max=input("Maximum Value in runs (e.g. replace above 100 with 100:")
		#cols = 1 if rets.ndim == 1 else rets.shape(1)
		if rets.ndim == 1 :	rets=rets.to_frame() 
		
		rands=np.random.randint(0,len(rets)-1,size=(Periods,Runs))
		
		def RunRand(eachret):
			randrets=pd.DataFrame(eachret.values[rands]+1)
			randcum=randrets.cumprod(axis=0)
			randdd=randcum/randrets.cummax()-1
			return pd.concat([randcum.iloc[-1].sort_values().reset_index(drop=True),randdd.min(axis=0).sort_values(ascending=False).reset_index(drop=True)],axis=1,keys=[eachret.name+" Cumdist",eachret.name +" DDdist"])
			
		allruns=pd.concat([RunRand(rets[x]) for x in rets.columns],axis=1)
		if allruns.ndim == 1 :	allruns=allruns.to_frame() 
		montecum=pd.DataFrame([allruns[x] for x in allruns.columns if "Cumdist" in x]).T
		montedd=pd.DataFrame([allruns[x] for x in allruns.columns if "DDdist" in x]).T
		
		#montecum=pd.DataFrame([pd.Series(randrets.prod(axis=0)-1).sort_values().reset_index(drop=True) for x in np.arange(rets.ndim)]).T
		#montecum.columns=rets.columns
		#montedd=pd.DataFrame([(pd.Series((randrets.cumprod()/np.maximum.accumulate(randrets)-1).min(axis=0))).sort_values().reset_index(drop=True) for x in np.arange(rets.ndim)]).T
		#montedd.columns=rets.columns
		#monte1[monte1<min]=min
		#monte1[monte1>max]=max
		
		return pd.concat([montecum.reset_index(drop=True),montedd.reset_index(drop=True)],axis=1,join_axes=[montecum.index])

	def SNR(self,prices,pers):
		'''Compute rolling X period Signal to Noise Ratio for each market'''
		return prices.rolling(pers).apply(lambda x: abs(x[-1]-x[0])/(abs(np.diff(x,axis=0)).sum(axis=0)))

	def AvgPairwiseCorr(self,rets):
		'''Average Pairwise Correlation across whole period'''
		allcor=rets.corr().stack()
		#for time series e.g. pandas Panel, use .to_frame().T and !=1
		#TimeSeriesOfAvgPairCorr=ACcorrpanels.to_frame().T[ACcorrpanels.to_frame().T!=1].mean(axis=1)
		return allcor[allcor!=1].mean()
		
	def AvgPairRollCorr(self,rets,*roll):
		'''Compute rolling average pairwise correlation across returns'''
		if len(roll) <1:
			roll=int(input("Rolling Period: "))
			allcorr=rets.rolling(int(roll)).corr().dropna()
		else:
			allcorr=rets.rolling(int(roll[0])).corr().dropna()
		
		return allcorr.to_frame().T[allcorr.to_frame().T.apply(lambda x: round(x,4))!=1].mean(axis=1)

	def RiskTarget(self,rets,cash):
		'''Transforms returns into risk targeted returns, with custom rebalancing period. Needs cash for leverage calcs'''
		#usecash = input("Use Cash Y/N? :")
		Annper = int(input("Days 260, Months 12 :"))
		target=float(input("Vol Target: "))
		rolls=int(input("Rolling Periods (520/24?): "))
		Rebal=input("Rebal Freq M/Q/Y? :")
		vols=rets.rolling(rolls).apply(lambda x:x.std()*np.sqrt(Annper)).dropna()
		#rebaldates=pd.date_range(start=vols.index[0],end=vols.index[-1],freq=Rebal)
		rebalwgt= vols.resample(Rebal).last().apply(lambda x: target/x).shift(1).dropna()
		rebalrets=rets.loc[rebalwgt.index[0]:]#(rets+1).resample(Rebal).prod()-1
		rebalcash=cash.loc[rebalwgt.index[0]:] #(cash+1).resample(Rebal).prod()-1
		cashlev=rebalwgt.apply(lambda x: 1-x)
		track = rebalrets*rebalwgt.loc[rebalrets.index].fillna(method="ffill") + rebalcash*cashlev.loc[rebalrets.index].fillna(method="ffill")
		
		return track

	def Summaryexport(self,rets,filename):
		pers=int(input("Monthly or Daily data? (12/252): "))
		stats=Stats(rets,1,pers)
		summary=rets.describe()
		annret=self.AnnualRet(rets)
		anndd=self.AnnualDD(rets)
		writer=pd.ExcelWriter(filename)
		stats.to_excel(writer,"Stats")
		summary.to_excel(writer,"Distribution")
		annret.to_excel(writer,"AnnualReturns")
		anndd.to_excel(writer,"AnnualDD")
		writer.save()
		print("Saved in %s" %filename + "\nStats, Distribution, AnnRet, AnnDD")
		
def StyleRolling(rets,per):
	ffdata=StyleReg(rets)[1]
	rf=ffdata["RF"].copy()
	percentret=ffdata.drop("RF",1)
	model=pd.ols(y=(rets-rf),x=percentret,window=per)
	betas=model.beta
	betas.columns=[x+" Coeff" for x in model.x.columns]
	tstats=model.t_stat
	tstats.columns=[x+" tstat" for x in model.x.columns]
	rsquared=model.r2
	return pd.concat([betas,tstats,rsquared.to_frame()],axis=1)
	
def Poisson(tosses,prob):
	from math import factorial
	return np.exp(-prob)*((prob**tosses)/factorial(tosses))
	
def StyleReg(rets):
	start=rets.index[0]
	end=rets.index[-1]
	region = input("US or Global returns? (US/Glbl)? ")
	if region == "US":
		ticker = ["F-F_Research_Data_5_Factors_2x3","F-F_Momentum_Factor"]
	else:
		ticker = ["Global_5_Factors","Global_Mom_Factor"]
	ffdata=pd.concat([pd.DataFrame(wb.DataReader(x,"famafrench",start,end)[0]) for x in ticker],axis=1).apply(lambda x: x/100)
	ffdata.index=ffdata.index.to_timestamp(freq="M")
	rf=ffdata["RF"].copy()
	percentret=ffdata.drop("RF",1)
	model=pd.ols(y=(rets-rf),x=percentret)
	res=model.summary_as_matrix
	res.loc["R Squared"] = model.r2
	return res.round(decimals=3), ffdata
	
def HistoricVolDist(filename):
	#C:\Users\ato\Documents\Ainsley Python\DATA\Compliance\S&P 1988.xlsx
	allret=pd.read_excel(filename)
	alldfs=[pd.DataFrame(allret.iloc[:,x]).set_index(allret.iloc[:,x-1]) for x in [1,3,5]]
	dret=alldfs[0].pct_change().dropna()
	wret=alldfs[1].pct_change().dropna()
	mret=alldfs[2].pct_change().dropna()
	drollvol=dret.rolling(252*5).apply(lambda x:x.std()*np.sqrt(252)).dropna()
	wrollvol=wret.rolling(52*5).apply(lambda x:x.std()*np.sqrt(52)).dropna()
	mrollvol=mret.rolling(12*5).apply(lambda x:x.std()*np.sqrt(12)).dropna()
	hists=pd.concat([x.iloc[:,0].value_counts(bins=[.00,.005,.02,.05,.1,.15,.25,.35]).sort_index().apply(lambda y: y/x.iloc[:,0].count()) for x in [drollvol,wrollvol,mrollvol]],axis=1)
		
	return pd.concat([dret.describe(),wret.describe(),mret.describe()],axis=1,keys=["daily","weekly","monthly"]),hists,[dret,wret,mret],[drollvol,wrollvol,mrollvol],print("RetDes","5yVolHist","List of Ret data","List of 5yVol Data")
	
def Corr100():
	zerovolret=np.random.normal(loc=0,scale=0.2/np.sqrt(252),size=252)
	posret=zerovolret+(.2/252)
	negret=zerovolret-(.2/252)
	return pd.DataFrame([posret,negret])
	
def BellCurve(runs):
	
	div = pd.Series(np.random.normal(loc=0, scale=0.16, size=(runs)))
	undiv = pd.Series(np.random.normal(loc=0, scale=0.32, size=(runs)))
	all=pd.DataFrame([div,undiv])
	#plt.hist(foo,histtype='step',bins=np.arange(undiv.min(),undiv.max(),.005))
	return all
	
def revrates(rets):
	newdates=pd.date_range(rets.index[-1],periods=len(rets))
	rev=rets.sort_index(ascending=false).reset_index(drop=True)
	rev.index=newdates
	return rev
	
def ALOadapt(percent):
	rainperc=percent*100
	rain=np.zeros(rainperc)
	sun=np.ones((100-rainperc))
	weather=np.concatenate([rain,sun])
	np.random.shuffle(weather)
	species=np.arange(0,1.25,0.25)
	env=weather[np.random.randint(0,100,size=20)]
	results=pd.DataFrame([np.where(env==1,x*3,(1-x)*3) for x in species]).T.cumprod()
	results.columns=species
	return results
	
def Putbigret(rets):
	rets.rolling(21).apply(lambda x: (1+x).prod()-1)
	rets["NegRet"].nlargest(30)
	return pd.concat([rolls["NegRet"].nlargest(30),rolls["S&PBlendRet"].loc[rolls["NegRet"].nlargest(30).index]],axis=1).to_excel("asda.xlsx")

def CAPErets(monthlyrets,CAPE):
	ret1=monthlyrets.apply(lambda x:x+1)
	roll1y = ret1.rolling(12).apply(lambda x: x.prod()-1).dropna()
	roll5y= ret1.rolling(60).apply(lambda x: (x.prod())**(1/5)-1).dropna()
	roll10y=ret1.rolling(120).apply(lambda x: (x.prod())**(1/10)-1).dropna()
	rolls=pd.concat([CAPE,roll1y,roll5y,roll10y],axis=1)
	rolls.columns=["CAPE","roll1y","roll5y","roll10y"]
	forwardrolls=pd.concat([rolls[x].dropna().reset_index(drop=True) for x in rolls.columns],axis=1)
	
	CAPEdeciles = forwardrolls.reset_index(drop=True).groupby(pd.Series(np.linspace(1,11,num=len(CAPE),dtype=int,endpoint=False)))
	CAPEdec=pd.concat([CAPEdeciles.min(),CAPEdeciles.mean(),CAPEdeciles.max()],axis=1)
	colnames=[forwardrolls.columns +" Min",forwardrolls.columns+" Avg",forwardrolls.columns + " Max"]
	CAPEdec.columns=[x for y in colnames for x in y]
	return rolls,forwardrolls,CAPEdec

def DrawdownVsVolExample(rets):
	#rets = pd.Series(np.random.normal(loc=.001,scale=.005,size=[260]))
	retsDD=rets.sort_values(rets.columns[0]).reset_index(drop=True)
	neg=retsDD[retsDD<=0]
	neg.index=np.arange(0,len(neg)*2,2)
	pos=retsDD[retsDD>0].sort_values(by=retsDD.columns[0],ascending=True)
	pos.index=np.arange(1,len(pos)*2,2)
	retsMinDD=pd.concat([neg,pos],axis=1).iloc[:,0].fillna(pos).reset_index(drop=True)
	#np.ravel(np.column_stack([neg,pos]))
	return pd.concat([rets,retsDD,retsMinDD],axis=1)
	
def FOMO():
	from AinsTools import cumulative,Drawdown
	Dret=pd.read_excel(r"C:\Users\ato\Documents\Ainsley Python\DATA\S&P Daily 1926 Ken French.xlsx",index_col="Date").iloc[:,0]
	DDD=Drawdown(cumulative(Dret))
	def res(pers):
		Xret=(1+Dret).resample(pers).prod()-1
		return Drawdown(cumulative(Xret))
	
	DDs=[DDD] + [res(x) for x in ["W-FRI","BM","BQ","BA"]] 
	DDtime=pd.Series([DDs[x][DDs[x]!=1].count()/DDs[x].count() for x in np.arange(5)],index=["D","W-FRI","M","Q","A"])
	
	#WDD,MDD,QDD,ADD = [DDs[x] for x in np.arange(4)]
	return pd.concat([DDs[x].resample("B").last().fillna(method="ffill") for x in np.arange(5)],axis=1,keys=["D","W-FRI","M","Q","A"]), DDtime

def IndustryPB():
	allindusPB=wb.DataReader("49_Industry_Portfolios","famafrench","31/12/1925")[6].apply(lambda x: 1/x)
	PBdeciles=pd.DataFrame([allindusPB[x].value_counts(bins=10).sort_index().reset_index(drop=True) for x in allindusPB.columns],index=allindusPB.columns)
	#PBdecvalues=pd.concat(pd.Series([allindusPB[x].value_counts(bins=10).sort_index().index.values).reset_index(drop=True) for x in allindusPB.columns],axis=0,keys=allindusPB.columns)
	#PBdecvalues.index = PBdecvalues.index +1
	#Currentdecile = pd.Series([PBdeciles[x][PBdecvalues[x] < allindusPB[x].iloc[-1]].max() for x in allindusPB.columns],index=allindusPB.columns)
	return allindusPB,PBdeciles,PBdecvalues,pd.concat([Currentdecile,PBdeciles.iloc[-1]],axis=1)

def SizeQuintiles():
	data=wb.DataReader("Portfolios_Formed_on_ME","famafrench","31/12/1920")[1]/100
	anns=AnnualRet(data)
	fig,ax=plt.subplots(5,2)
	[ax[x].bar(np.arange(5),anns.tail(10).iloc[x,4:8]) for x in range(10)]
	[ax[x].set_title(anns.tail(10).index[x]) for x in range(10)]
	[ax[x].set_xticklabels([x for x in anns.columns[4:8]]) for x in range(10)]
	return anns
	
def AQRData():
	'''Pull data from AQR QMJ'''
	AQRsheet=pd.ExcelFile("https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx")
	Mkt,SMB,HMLdev,UMD,RF,QMJ=[AQRsheet.parse(x,header=18,index_col=0) for x in ["MKT","SMB","HML Devil","UMD","RF","QMJ Factors"]]
	return dict(list(zip(["MKT","SMB","HML Devil","UMD","RF","QMJ Factors"],[Mkt,SMB,HMLdev,UMD,RF,QMJ])))
	
def ShillerData():
	'''Pull data from shiller data and calculate monthly equity and 10y bond TRs, and then real annual price returns including real estate'''
	shillerdata=pd.read_excel("http://www.econ.yale.edu/~shiller/data/ie_data.xls",sheet_name="Data",header=7)

	data=shillerdata.iloc[:-2].copy()
	dates=pd.date_range(start="31/1/1871",periods=len(data),freq="M")
	data.index=dates
	rates=data["Rate GS10"]/100
	yearendcoupon=data["Rate GS10"].groupby(dates.year).last()
	startmonthbondprice=pd.Series(np.where(dates.month==1,100,
									np.pv(
									rate=rates.shift(1),#ending rate 1 period prior as this is start of month bond price
									nper=10,
									pmt=-yearendcoupon.shift(1).loc[dates.shift(-1).year],#this is starting coupon on a bond from the start of the previous period (e.g. at the start of Feb, this is the value of a coupon received during Jan, which was the yield at the end of Dec)
									fv=-100,
									when="end"
									)
									),index=dates)
	
	#one line version : startmonthbondprice=pd.Series(np.where(dates.month==1,100,np.pv(rate=rates.shift(1),nper=10,pmt=-yearendcoupon.shift(1).loc[dates.shift(-1).year],fv=-100,when="end")),index=dates)
	endmonthbondprice=pd.Series(
								np.pv(
										rate=rates,
										nper=10,
										pmt=-yearendcoupon.shift(1).loc[dates.year],
										fv=-100,
										when="end"
										),
										index=dates)
	
	#one line version: endmonthbondprice=pd.Series(np.pv(rate=rates,nper=10,pmt=-yearendcoupon.shift(1).loc[dates.year],fv=-100,when="end"),index=dates)
	bondpriceret=endmonthbondprice/startmonthbondprice-1
	bondincret=yearendcoupon.shift(1).loc[dates.year]/1200
	bondincret.index=dates
	
	bondtr=bondpriceret+bondincret
	d12=data["D"]/12
	equitytr=(data["P"]+d12)/data["P"].shift(1)-1
	
	def annLT():
		'''Create Annual Data Series from CaseShiller data sheet, 
		from mix of annual and monthly (1953-) and match to eq and bond annual'''
		redata=pd.read_excel("http://www.econ.yale.edu/~shiller/data/Fig3-1.xls",sheet_name="Data",header=6)
		redates=redata.iloc[:,0].dropna()
		reprm=pd.Series(redata.iloc[:,8].dropna().values,index=redates.apply(lambda x:int(x)).values)
		annre=reprm.groupby(reprm.index).last()
		annre.index=pd.date_range(start=str(annre.index[0]),end=str(annre.index[-1]+1),freq="A")
		anneq=data["P"].resample("A").last()
		annbd=(1+bondpriceret).cumprod().resample("A").last()
		fin=pd.concat([anneq,annbd,annre],axis=1).dropna()
		fin.columns=["EqNomPrice","BdNomPrice","PropNomPrice"]
		return fin

	annprice=annLT()
	
	return equitytr,bondtr,annprice.pct_change().dropna()

def ShillerPlot(xs,ys):
	fix,ax = plt.subplots()
	ax.bar(xs,ys.iloc[:,0],color="b",align="center")
	ax.bar(xs,ys.iloc[:,1],color="g",align="center")
	ax.xaxis.set_ticks(np.arange(min(xs.index),max(xs.index)+1,10))
	plt.show()
	
def DataCheck(rets):
	freqs=input("Frequency of data (D/W/M/Q)? :")
	if freqs=="D":
		bfreqs="B"
	elif freqs=="W":
		bfreqs="W-FRI"
	else:
		bfreqs="B"+freqs
		
	busdates=pd.date_range(start=rets.index[0],end=rets.index[-1],freq=bfreqs)
	dates=pd.date_range(start=rets.index[0],end=rets.index[-1],freq=freqs)
	bmiss=rets.index[busdates].isnull().sum()
	cmiss=rets.index[dates].isnull().sum()
	percnan=rets.isnull().sum()/rets.count()
	return pd.Series([bmiss,cmiss,percnan])

def EmailFile(TOadd,Subj,Messg,Attch,**kwargs):
	'''Uses win32com to send file'''
	import win32com.client as win32
	outlook = win32.Dispatch("Outlook.Application")
	mail=outlook.CreateItem(0)
	mail.To=TOadd
	mail.Subject=Subj
	mail.Body=Messg
	#if sending df, senddf=df.to_html()
	#mail.HTMLBody=senddf
	if len(kwargs) > 0:
		mail.HTMLBody = kwargs["HTML"]
	mail.Attachments.Add(Attch)
	mail.Send()
	
	
if __name__ == "__main__":
	if len(sys.argv) >1:
		file = sys.argv[1]
		
		if len(sys.argv)>2:
			data =pd.read_excel(file,sheetname=sys.argv[2],index_col="Date")
		else:	
			data =pd.read_excel(file,index_col="Date")
		
		#plot(colormap=cm.get_cmap("hsv"))
		
	pdb.set_trace()
#data=pd.read_excel(r"C:\Users\ato\Documents\Ainsley Python\DATA\AQR\AQR Trend Daily Sep2016.xlsx")
	#data=pd.read_excel(r"C:\Users\ato\Documents\Ainsley Python\lt returns\DATA\LT Asset Class Returns.xlsx")
	#cums=pd.read_excel(r"C:\Users\ato\Documents\Ainsley Python\DATA\GS Trend\GSAM Recon.xlsx")
	#LTsharpesTFlev=(cumTFlev.iloc[-1]**(12/len(cumTFlev))-1-)/(TFlev.std()*np.sqrt(12))
