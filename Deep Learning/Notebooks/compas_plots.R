

#####
## January 11, 2019
## Create figures from results in BNN COMPAS.ipynb.
## 
## Should have COMPAS_BNN_RATE.csv and 
##    COMPAS_BNN_RATE_multinomialResponse.csv saved to file.
## The former is `rate_data_csv` in the Jupyter notebook from running all cells twice: 
##  once with all variables, once with num. priors omitted, both times with multiclass flag = False.
## The latter is `rate_data_csv` in the Jupyter notebook, generated the same way but with 
##  multiclass flag = True for both runs.
#####

library(gridExtra)
library(onehot)
library(AUC)
library(data.table)
library(dplyr)
library(ggplot2)
library(gtools)
library(plyr)

# --- Delta and ESS function
# num.pred: number of predictors
# length.nullify: number of predictors we conditioned on as zero
# num.pred is 10 and length.nullify is 0 for all except 
#     the second-order rate values, which has length.nullify=1
info.func <- function(rate, num.pred=p, length.nullify=0) {
  ### Find the entropic deviation from a uniform distribution ###
  delta <- sum(rate*log((num.pred-length.nullify)*rate)) 
  
  ### Calibrate Delta via the effective sample size (ESS) measures from importance sampling ###
  #(Gruber and West, 2016, 2017)
  ess <- 1/(1+delta)*100
  
  return(list('delta'=delta, 'ess'=ess))
}

# --- Get compas data
raw_data <- fread("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv", 
                  data.table=FALSE)

# duplicate data columns
raw_data <- raw_data[, -which(duplicated(names(raw_data)))]

# --- choose variables and remove missingness the way propublica does
# 6172 obs.
nomissing.df <- raw_data %>% dplyr::select(age, c_charge_degree, race, age_cat, 
                                           score_text, 
                                           sex, priors_count, 
                                           days_b_screening_arrest, decile_score, is_recid, 
                                           two_year_recid, c_jail_in, c_jail_out) %>% 
  filter(days_b_screening_arrest <= 30) %>%
  filter(days_b_screening_arrest >= -30) %>%
  filter(is_recid != -1) %>%
  filter(c_charge_degree != "O") %>%
  filter(score_text != 'N/A')

nomissing.df['Race'] <- ifelse(nomissing.df$race == "Caucasian", "White", "Non-White")

# --- RATE values from BNN - Binary (main text of paper)
# includes first order centrality, second order centrality (num priors --> null) 
# and first order centrality when num priors is scrambled
# This is `rate_data_csv` in the Jupyter notebook, created after running the notebook cells 
#   once with all variables and a second time with num. priors omitted.
rate.dat <- read.csv('COMPAS_BNN_RATE.csv')
rate.dat[rate.dat$var_names=="priors_count", 
         grepl("2ndOrder", names(rate.dat))] <- NA  # 2nd order centrality is NA, not 0, for nullified var

p <- nrow(rate.dat)  # num predictors


# --- RATE values from BNN - Multiclass (supplemental)
# includes first order for all three classes, 
# second order for all three classes, 
# first order for all three classes when num. priors is omitted, 
# second order for all three classes when num. priors is omitted
# Like the data above, created from running the Jupyter notebook twice (but with multiclass flag = True)
rate.dat.multi <- read.csv('COMPAS_BNN_RATE_multinomialResponse.csv')




# --- Plot 1
# first order bar chart
num.nullify = 0
info <- info.func(rate.dat$rate_class0_1stOrder, num.pred=p, length.nullify=num.nullify )
DELTA = round(info$delta, 3)
ESS = round(info$ess, 3)

# we'll retain this ordering for all plots
rate.dat$feature <- factor(rate.dat$feature, 
                           levels = rate.dat$feature[order(rate.dat$rate_class0_1stOrder, 
                                                           decreasing=TRUE)])

plt <- ggplot(rate.dat, aes(x = feature, y = rate_class0_1stOrder)) + 
  geom_bar(stat = "identity", fill = "grey80") + 
  # remove background
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) + 
  # rotate labels
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  # remove ticks on x axis 
  theme(axis.ticks.x=element_blank()) + 
  # axis titles
  labs(x = "Covariates", 
       y = expression(RATE(tilde(beta)[j])), 
       title = "First Order Centrality") + 
  # center title 
  theme(plot.title = element_text(hjust = 0.5)) +  
  # title text size
  theme(plot.title = element_text(size = 9)) + 
  # add line at 1/p
  geom_hline(yintercept = 1/p, lty="dashed", col="red") + 
  # ensure (0, 1) limits for y-axis
  scale_y_continuous(limits = c(0, 1)) + 
  # ESS AND DELTA
  annotate("text", x = c(7.3, 7.5, 7.3, 7.5), 
           y = c(0.95, 0.95, 0.9, 0.9), 
           size=c(6, 3, 6, 3), 
           hjust = c(0, 0, 0, 0), 
           label = c(".", paste0("DELTA = ", DELTA), 
                     ".", paste0("ESS = ", ESS)), 
           col = c("red", "black", "red", "black"))

plt







# --- Plot 2
# 2 histograms overlaying each other
# white v. not-white
# of prior counts
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

# p-value on KS test for difference of priors count densities between white / non-white
pval <- ks.test(nomissing.df[nomissing.df$Race == "White", 'priors_count'], 
                nomissing.df[nomissing.df$Race == "Non-White", 'priors_count'])[["p.value"]]

# define info text
info_text <- c(paste0("`KS Test P-Value`", "%~~%", (pval)), 
               paste0("\nNum.~Non-White:~", 
                      (table(nomissing.df$Race)[["Non-White"]])), 
               paste0("\nNum.~White:~", 
                      (table(nomissing.df$Race)[["White"]])))

plt <- ggplot(nomissing.df) + 
  geom_density(aes(x = priors_count, group=Race, fill=Race), 
               alpha=0.5, adjust=4) + 
  # remove background
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) + 
  # axis titles
  labs(x = "Number of Prior Offenses", 
       y = "Density", 
       title = "Number of Prior Offenses by Racial Group", 
       fill = "Racial Group") + 
  # title text size
  theme(plot.title = element_text(size = 9)) + 
  # legend text size
  theme(legend.title = element_text(size = 9)) + 
  # center title 
  theme(plot.title = element_text(hjust = 0.5)) + 
  # fill densities
  scale_fill_manual(values=c(cbPalette[1], cbPalette[3])) + 
  # set y-axis to maximum density (manual selection.. be careful)
  # and get rid of weird empty space between axis and density line
  scale_y_continuous(limits = c(0, 0.2), expand = c(0, 0)) + 
  # set x-axis scale to maximum priors_count 
  # and get rid of weird empty space between axis and density line
  scale_x_continuous(limits = c(0, max(nomissing.df$priors_count)), 
                     expand = c(0, 0)) + 
  # add text with num. non-white, num. white and k-s test p-value
  annotate("text", 
           x = c(20, 20, 20), 
           y = c(0.14, 0.13, 0.12), 
           size = c(3, 3, 3), hjust = c(0, 0, 0),
           label = info_text, 
           parse = TRUE) + 
  # Remove the plot legend
  theme(legend.position='none') + 
  # add in manual legend that goes with text annotation
  annotate("text", 
           x = c(20, 20, 20, 20, 20), 
           y = c(0.182, 0.175, 0.175, 0.17, 0.17), 
           size = c(3, 8, 3, 8, 3), hjust = rep(0, 5), 
           label = c("Racial Group", 
                     ".", 
                     "    Non-White", 
                     ".", 
                     "    White"), 
           col = c("black", cbPalette[1], "black", cbPalette[3], "black"))

plt




# --- Plot 3
# first order bar chart, but num priors is omitted from analysis

rate.dat.temp <- rate.dat[rate.dat$var_names != "priors_count", ]

num.nullify = 0
info <- info.func(rate.dat.temp$rate_class0_1stOrder_numPriorsOmitted, 
                  num.pred=p, length.nullify=num.nullify )
DELTA = round(info$delta, 3)
ESS = round(info$ess, 3)

rate.dat.temp$feature <- factor(rate.dat.temp$feature, 
                                levels = 
                                  levels(rate.dat$feature)[-which(levels(rate.dat$feature) == "Num. Priors")])


plt <- ggplot(rate.dat.temp, aes(x = feature, y = rate_class0_1stOrder_numPriorsOmitted)) + 
  geom_bar(stat = "identity", fill = "grey80") + 
  # remove background
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) + 
  # rotate labels
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  # remove ticks on x axis 
  theme(axis.ticks.x=element_blank()) + 
  # axis titles
  labs(x = "Covariates", 
       y = expression(RATE(tilde(beta)[j])), 
       title = "First Order Centrality\nPrior Offenses Omitted from Data") + 
  # center title 
  theme(plot.title = element_text(hjust = 0.5)) + 
  # title text size
  theme(plot.title = element_text(size = 9)) + 
  # add line at 1/p
  geom_hline(yintercept = 1/p, lty="dashed", col="red") + 
  # ensure (0, 1) limits for y-axis
  scale_y_continuous(limits = c(0, 1)) + 
  # ESS AND DELTA
  annotate("text", x = c(7.3, 7.5, 7.3, 7.5), 
           y = c(0.95, 0.95, 0.9, 0.9), 
           size=c(6, 3, 6, 3), 
           hjust = c(0, 0, 0, 0), 
           label = c(".", paste0("DELTA = ", DELTA), 
                     ".", paste0("ESS = ", ESS)), 
           col = c("red", "black", "red", "black"))
plt






# --- Plot 1 FOR SUPPLEMENTALS
# first order bar chart for Low / first order bars for Medium / First Order bars for High
num.nullify = 0
info <- sapply(0:2, function(response) {
  info.func(rate.dat.multi[, paste0("rate_class", response, "_1stOrder")], 
                  num.pred=p, 
                  length.nullify=num.nullify)
})
DELTA = round(unlist(info["delta", ]), 3)
ESS = round(unlist(info["ess", ]), 3)

first_order_columns <- c("rate_class0_1stOrder", "rate_class1_1stOrder", "rate_class2_1stOrder")
  
# we'll retain this ordering for all plots
rate.dat.multi$feature <- factor(rate.dat$feature, 
                           levels = rate.dat$feature[order(rate.dat$rate_class0_1stOrder, 
                                                           decreasing=TRUE)])

rate.dat.multi.temp <- rate.dat.multi
# format to look like the edits Lorin manually gave the main text plots
rate.dat.multi.temp$feature <- 
  revalue(rate.dat.multi.temp$feature, c("Race: African-American" = "Race: AA", 
                                         "Race: Native-American" = "Race: NA"))


for (response in 1:3) {
  assign(paste0("plt", response), 
         ggplot(rate.dat.multi.temp, aes_string(x = "feature", 
                                           y = first_order_columns[response])) + 
            geom_bar(stat = "identity", fill = "grey80") + 
            # remove background
            theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                  panel.background = element_blank(), axis.line = element_line(colour = "black")) + 
            # rotate labels
            theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
            # remove ticks on x axis 
            theme(axis.ticks.x=element_blank()) + 
            # axis titles
            labs(x = "", 
                 y = ifelse(response == 1, "RATE", ""), 
                 title = paste0(c("Low", "Medium", "High")[response], " Risk")) + 
            # center title 
            theme(plot.title = element_text(hjust = 0.5)) +  
            # title text size
            theme(plot.title = element_text(size = 9)) + 
            # add line at 1/p
            geom_hline(yintercept = 1/p, lty="dashed", col="red") + 
            # ensure (0, 1) limits for y-axis
            scale_y_continuous(limits = c(0, 1)) + 
            # ESS AND DELTA
            annotate("text", x = c(4.4, 4.9, 4.4, 4.9), 
                     y = c(0.95, 0.95, 0.92, 0.92), 
                     size=c(6, 3, 6, 3), 
                     hjust = c(0, 0, 0, 0), 
                     label = c(".", paste0("DELTA = ", DELTA[response]), 
                               ".", paste0("ESS = ", ESS[response])), 
                     col = c("red", "black", "red", "black"))
  )
}

grid.arrange(plt1, plt2, plt3, ncol=3)







# --- Plot 2 FOR SUPPLEMENTALS
# first order centrality WHEN NUM. PRIORS IS OMITTED
# first order bar chart for Low / first order bars for Medium / First Order bars for High

rate.dat.multi.temp <- rate.dat.multi[-which(rate.dat.multi$var_names == 'priors_count'), ]

num.nullify = 0
info <- sapply(0:2, function(response) {
  info.func(rate.dat.multi.temp[, paste0("rate_class", response, "_1stOrder_numPriorsOmitted")], 
            num.pred=p, 
            length.nullify=num.nullify)
})
DELTA = round(unlist(info["delta", ]), 3)
ESS = round(unlist(info["ess", ]), 3)

first_order_columns <- c("rate_class0_1stOrder_numPriorsOmitted", 
                         "rate_class1_1stOrder_numPriorsOmitted", 
                         "rate_class2_1stOrder_numPriorsOmitted")

rate.dat.multi.temp$feature <- factor(rate.dat.multi.temp$feature, 
                                levels = 
                                  levels(rate.dat.multi$feature)[-which(levels(rate.dat.multi$feature) == "Num. Priors")])

# format to look like the edits Lorin manually gave the main text plots
rate.dat.multi.temp$feature <- 
  revalue(rate.dat.multi.temp$feature, c("Race: African-American" = "Race: AA", 
                               "Race: Native-American" = "Race: NA"))

for (response in 1:3) {
  assign(paste0("plt", response), 
         ggplot(rate.dat.multi.temp, aes_string(x = "feature", 
                                           y = first_order_columns[response])) + 
           geom_bar(stat = "identity", fill = "grey80", width=0.8) + 
           # remove background
           theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                 panel.background = element_blank(), axis.line = element_line(colour = "black")) + 
           # rotate labels
           theme(axis.text.x = element_text(angle = 55, hjust = 1)) + 
           # remove ticks on x axis 
           theme(axis.ticks.x=element_blank()) + 
           # axis titles
           labs(x = "", 
                y = ifelse(response == 1, "RATE", ""), 
                title = paste0(c("Low", "Medium", "High")[response], " Risk")) + 
           # center title 
           theme(plot.title = element_text(hjust = 0.5)) +  
           # title text size
           theme(plot.title = element_text(size = 9)) + 
           # add line at 1/p
           geom_hline(yintercept = 1/p, lty="dashed", col="red") + 
           # ensure (0, 1) limits for y-axis
           scale_y_continuous(limits = c(0, 1)) + 
           # ESS AND DELTA
           annotate("text", x = c(3.4, 3.9, 3.4, 3.9), 
                    y = c(0.95, 0.95, 0.92, 0.92), 
                    size=c(6, 3, 6, 3), 
                    hjust = c(0, 0, 0, 0), 
                    label = c(".", paste0("DELTA = ", DELTA[response]), 
                              ".", paste0("ESS = ", ESS[response])), 
                    col = c("red", "black", "red", "black"))
  )
}

grid.arrange(plt1, plt2, plt3, ncol=3)

















