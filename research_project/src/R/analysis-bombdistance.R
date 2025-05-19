#' Analysis script for hypothesis testing CS:GO demo replay data.
#' Focus is always the T-side.
#' 
#' Notes: 1) seconds and clocktime are resetted after bomb plant.
#'        2) tbd.
#' 
#' @author Joschka A. Huellmann <j.huellmann@utwente.nl>
#' @date 2024-10-15
#'

library(dplyr)

setwd("..\\src\\")

#### PREPROCESS DATA FRAME ####
df <- read.csv("testdemo6.csv")
df$afterAdverseEvent = df$t_alivePlayers < 5
df$bombPlanted = as.logical(df$bombPlanted)
df$tDmg = 500 - df$ctHp


#### DESCRIPTIVE ANALYSIS ####

# round lengths
barplot(maxRoundLength ~ roundNum, df %>% group_by(roundNum) %>% summarise(maxRoundLength = max(secondsCalculated)) %>% data.frame)

# minimum equipment both teams per round
barplot(minEq ~ roundNum, df %>% group_by(roundNum) %>% summarise(minEq = min(t_teamEqVal, ct_teamEqVal)) %>% data.frame)

# ratio of rounds with bomb planted
df %>% group_by(roundNum) %>% count(bombPlanted)

# plot positions
df.test <- df %>% filter(roundNum == 1)
plot(c(df.test$t1_x,df.test$t2_x,df.test$t3_x,df.test$t4_x,df.test$t5_x), c(df.test$t1_y,df.test$t2_y,df.test$t3_y,df.test$t4_y,df.test$t5_y))

# plot bombdistance, mapcontrol, hp, playeralive, velocityDeviation
df.test <- df %>% filter(roundNum == 3) %>%
	select( velocityDeviation,tHp,tDmg,ctHp,bombDistance,mapControl,deltaDistance,totalDistance,
			secondsCalculated,afterAdverseEvent,bombPlanted,bombSeconds,
			t1_x,t2_x,t3_x,t4_x,t5_x,t1_y,t2_y,t3_y,t4_y,t5_y)
timeAdverseEvent <- df %>% filter(roundNum == 3, afterAdverseEvent == TRUE) %>% pull(secondsCalculated) %>% first()
par(mfrow=c(2,4))
plot(df.test$secondsCalculated, df.test$tHp, ylim=c(0,500), ylab = "Team T HP", xlab = "Seconds", col = rgb(df.test$afterAdverseEvent,0,df.test$bombPlanted))
abline(v = df.test$bombSeconds, lty="dashed")
abline(v = timeAdverseEvent)
plot(df.test$secondsCalculated, df.test$tDmg, ylim=c(0,500), ylab = "Team T Damage", xlab = "Seconds", col = rgb(df.test$afterAdverseEvent,0,df.test$bombPlanted))
abline(v = df.test$bombSeconds, lty="dashed")
abline(v = timeAdverseEvent)
plot(c(df.test$t1_x,df.test$t2_x,df.test$t3_x,df.test$t4_x,df.test$t5_x), c(df.test$t1_y,df.test$t2_y,df.test$t3_y,df.test$t4_y,df.test$t5_y),
	xlim = c(-2200,1800), ylim=c(-1200,3200), ylab = "Y Positions", xlab = "X Positions", col = rgb(df.test$afterAdverseEvent,0,df.test$bombPlanted))
# no ablines
plot(df.test$secondsCalculated, df.test$deltaDistance, ylim=c(0,1000), ylab = "Team T Delta Distance", xlab = "Seconds", col = rgb(df.test$afterAdverseEvent,0,df.test$bombPlanted))
abline(v = df.test$bombSeconds, lty="dashed")
abline(v = timeAdverseEvent)
plot(df.test$secondsCalculated, df.test$totalDistance, ylim=c(0,60000), ylab = "Team T Total Distance", xlab = "Seconds", col = rgb(df.test$afterAdverseEvent,0,df.test$bombPlanted))
abline(v = df.test$bombSeconds, lty="dashed")
abline(v = timeAdverseEvent)
plot(df.test$secondsCalculated, df.test$mapControl, ylim=c(-1,1), ylab = "Map Control", xlab = "Seconds", col = rgb(df.test$afterAdverseEvent,0,df.test$bombPlanted))
abline(v = df.test$bombSeconds, lty="dashed")
abline(v = timeAdverseEvent)
plot(df.test$secondsCalculated, df.test$bombDistance, ylim=c(0,4800), ylab = "Distance Bomb to Bombsite", xlab = "Seconds", col = rgb(df.test$afterAdverseEvent,0,df.test$bombPlanted))
abline(v = df.test$bombSeconds, lty="dashed")
abline(v = timeAdverseEvent)
plot(df.test$secondsCalculated, df.test$velocityDeviation, ylim=c(0,210), ylab = "Velocity Deviation", xlab = "Seconds", col = rgb(df.test$afterAdverseEvent,0,df.test$bombPlanted))
abline(v = df.test$bombSeconds, lty="dashed")
abline(v = timeAdverseEvent)
par(mfrow=c(1,1))



#### TESTING ####

# Test that estimated times are correct
df.test <- df %>% filter(roundNum == 5) %>% mutate(secondsDiff = secondsCalculated - seconds) %>% select(tick, seconds, secondsCalculated, secondsDiff)

# Get map boundaries and other metric boundaries
# [1] -2186 -> -2200
df.test <- df
min(c(df.test$t1_x,df.test$t2_x,df.test$t3_x,df.test$t4_x,df.test$t5_x,df.test$ct1_x,df.test$ct2_x,df.test$ct3_x,df.test$ct4_x,df.test$ct5_x))
# [1] 1788 -> 1800
max(c(df.test$t1_x,df.test$t2_x,df.test$t3_x,df.test$t4_x,df.test$t5_x,df.test$ct1_x,df.test$ct2_x,df.test$ct3_x,df.test$ct4_x,df.test$ct5_x))
# [1] -1163.997 -> -1200
min(c(df.test$t1_y,df.test$t2_y,df.test$t3_y,df.test$t4_y,df.test$t5_y,df.test$ct1_y,df.test$ct2_y,df.test$ct3_y,df.test$ct4_y,df.test$ct5_y))
# [1] 3118 -> 3200
max(c(df.test$t1_y,df.test$t2_y,df.test$t3_y,df.test$t4_y,df.test$t5_y,df.test$ct1_y,df.test$ct2_y,df.test$ct3_y,df.test$ct4_y,df.test$ct5_y))
# [1] 58568.03 -> 60000
max(df.test$totalDistance)
# [1] 937.8394 -> 1000
max(df.test$deltaDistance)
# [1] 203.9484 -> 210
max(df.test$velocityDeviation)
# [1] 4575.033 -> 4800
max(df$bombDistance, na.rm = TRUE)






df <- read.csv("testdemo5.csv")
df.twin <- df %>% filter(winningSide == "T")
df.tlose <- df %>% filter(winningSide == "CT")