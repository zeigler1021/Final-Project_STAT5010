# ## Graveyard 
# 
# Trying a test-train model? eh 
# ```{r}
# daily_data |>
#   pivot_longer(3:12) |>
#   select(-mins_daily) |>
#   ggplot(aes(x = steps_daily, y = value)) +
#   geom_point() +
#   facet_wrap(~name)
# 
# #I want to predict how much I'll walk based on how much music I listen to
# train <- slice_sample(daily_data, prop = 0.8) 
# test <- anti_join(daily_data, train)
# 
# linreg <- lm(steps_daily ~ mins_daily, train)
# plot(linreg)
# summary(linreg)
# 
# 
# ggplot(daily_data, aes(x = mins_daily, y = steps_daily)) +
#   geom_point() +
#   geom_smooth(method = "lm")
# 
# summary(lm(steps_daily ~ mins_daily, daily_data))
# summary(lm(snow ~ ., daily_data))
# 
# ```
# 
# Does any audio feature do a better job at predicting the # of steps I'll take?
# ```{r}
# test <- daily_data |>
#   select(3:13) 
# 
# full <- lm(steps_daily ~ ., data = test)
# red <- lm(steps_daily ~ danceability_daily + tempo_norm_daily, data = test)
# 
# anova(red, full)
# 
# #I need the full model?!?!?!?!
# ```
# 
# 
# 
# ```{r}
# ggplot(daily_data, aes(x = date, y = mins_daily)) +
#   geom_line()
# 
# ggplot(daily_data, aes(x = date, y = steps_daily)) +
#   geom_line()
# 
# ggplot(daily_data, aes(x = tmax, y = energy_daily)) +
#   geom_line()
# 
# ggplot(daily_data, aes(x = danceability_daily, y = energy_daily)) + 
#   geom_point() + 
#   geom_smooth(method = "lm") 
# 
# ggplot(daily_data, aes(x = danceability_daily, y = valence_daily)) + 
#   geom_point() + 
#   geom_smooth(method = "lm")
# 
# ggplot(daily_data, aes(x = danceability_daily, y = steps_daily)) + 
#   geom_point() + 
#   geom_smooth(method = "lm")
# 
# 
# ggplot(daily_data, aes(x = steps_daily, y = energy_daily)) + 
#   geom_point()
# 
# ggplot(daily_data, aes(x = tmean, y = steps_daily)) + 
#   geom_point()
# 
# ggplot(spotify, aes(x = energy, y = instrumentalness, color = high_deviation)) +
#   geom_point()
# ```
# 
# ```{r}
# #This reveals nothing bc of te way I defined "high deviation"
# test <- daily_data |>
#   pivot_longer(3:12) |>
#   ggplot(aes(x = weather_desc, y = value)) +
#   geom_boxplot() +
#   facet_wrap(~name)
# test
# ```
# 
# Exploring Deviations:
#   ```{r}
# spotify |>
#   pivot_longer(c(29:32,12,13,17:21)) |>
#   ggplot(mapping = aes(y = value, color = high_deviation)) +
#   geom_boxplot() +
#   facet_wrap(~name)
# 
# summary(lm(loudness_norm ~ high_deviation, spotify))
# summary(lm(time_signature_norm ~ high_deviation, spotify))
# summary(lm(instrumentalness ~ high_deviation, spotify))
# summary(lm(speechiness ~ high_deviation, spotify))
# summary(lm(energy ~ high_deviation, spotify))
# summary(lm(liveness ~ high_deviation, spotify))
# 
# ```
# 
# Because of the way I encoded 'deviations', this dataset is not super useful and unfortunately I'm too tired to figure out how to make it better. :^)
# 
# ```{r}
# performance::model_performance(lmod_steps)
# performance::model_performance(lm(steps_daily ~ . - tempo_norm_daily, data = steps_daily))
# performance::model_performance(lm(steps_daily ~ . - tempo_norm_daily - valence_daily, data = steps_daily))
# performance::model_performance(lm(steps_daily ~ . - tempo_norm_daily - valence_daily - liveness_daily, data = steps_daily))
# performance::model_performance(lm(steps_daily ~ . - tempo_norm_daily - valence_daily - liveness_daily - speechiness_daily, data = steps_daily))
# performance::model_performance(lm(steps_daily ~ . - tempo_norm_daily - valence_daily - liveness_daily - speechiness_daily - energy_daily, data = steps_daily))
# performance::model_performance(lm(steps_daily ~ . - tempo_norm_daily - valence_daily - liveness_daily - speechiness_daily - energy_daily - danceability_daily, data = steps_daily))
# performance::model_performance(lm(steps_daily ~ . - tempo_norm_daily - valence_daily - liveness_daily - speechiness_daily - energy_daily - danceability_daily - mins_daily, data = steps_daily))
# ```
# 
# Transforming my data?
# ```{r}
# summary(lm(log(tmean) ~ log(steps_daily), data = daily_data))
# summary(lm(log(energy_daily)~log(steps_daily), data = daily_data))
# summary(lm(steps_daily ~ energy_daily, daily_data))
# summary(lm(steps_daily ~ instrumentalness_daily, daily_data))
# summary(lm(steps_daily ~ acousticness_daily, daily_data))
# 
# ```
# 
# Time Series Shite 
# ```{r}
# daily_data_ts <- daily_data |>
#   select(-c(steps_desc, weather_desc))
# daily_data_ts <- as.xts(read.zoo(daily_data_ts, format = "%Y-%m-%d"))
# plot(daily_data_ts, legend.loc = "topleft")
# 
# audio_features <- daily_data |> 
#   select(-c(mins_daily, steps_daily, tmax, tmin, tmean, precip, snow, snowcover, steps_desc, weather_desc)) 
# 
# audio_features_ts <- as.xts(read.zoo(audio_features, format = "%Y-%m-%d"))
# 
# 
# 
# plot(rollmean(daily_data_ts, 7, align = "right"))
# 
# plot(audio_features_ts, multi.panel = TRUE)
# plot(rollmean(audio_features_ts, 7, align = "right"))
# plot(rollmean(audio_features_ts, 7), multi.panel = TRUE)
# 
# ```
# 
# ANOVA--
# ```{r}
# summary(lm(steps_daily ~ , daily_data))
# ```
# 
# 
# ```{r}
# summary(lm(valence_daily ~ weather_desc, data = daily_data))
# summary(lm(energy_daily ~ weather_desc, data = daily_data))
# summary(lm(mins_daily ~ weather_desc, data= daily_data))
# summary(lm(acousticness_daily ~ weather_desc, data= daily_data))
# 
# 
# summary(lm(mins_daily ~ steps_desc, daily_data))
# summary(lm(energy_daily ~ steps_desc, daily_data))
# summary(lm(danceability_daily ~ steps_desc, daily_data))
# summary(lm(instrumentalness_daily ~ steps_desc, data= daily_data))
# 
# 
# plot(lm(danceability_daily ~ steps_desc, daily_data))
# 
# 
# ggplot(daily_data, aes(x = steps_daily, y = danceability_daily, color = steps_desc)) +
#   geom_point() +
#   geom_smooth(method = "lm")
# ```
# 
# 
# Defining Music Taste:
# ```{r deviations}
# # Here I define what constitutes a 'high deviation track'. It is when the audio feature has a value outside ± 2*sd of that audio feature. 
# high_deviation <- spotify |>
#   mutate(row = row_number()) |>
#   pivot_longer(cols = c(danceability, energy, loudness_norm, valence, tempo_norm, acousticness, instrumentalness, speechiness)) |>
#     group_by(name) |>
#   mutate(mean_feature = mean(value), 
#             sd_feature = sd(value)) |>
#   mutate(high_deviation = case_when(value >= (mean_feature + 2*sd_feature) | value <= (mean_feature) - 2*sd_feature ~ TRUE,
#                                     TRUE ~ FALSE))
# 
# #Here, I set up this dataframe to be mapped
# high_deviation <- high_deviation |> 
#   select(row, high_deviation) |>
#   group_by(row) |>
#   nest()
# # Here, I have to determine if a single track has high deviation or not. I based that on if ANY single audio feature deviates. If a single audio feature for a track deviates, I mark the track as "high deviation"
# high_deviation <- map(high_deviation[[2]], determine.deviation) |>
#   unlist()
# 
# #Here, I add the newly mapped 'high deviation' vector to my spotify data so I can use it! 
# spotify <- bind_cols(spotify, high_deviation = high_deviation)
# ```
# 
# Creating factors to describe weather/activity data:
# ```{r}
# daily_data <- daily_data |> 
#   mutate(weather_desc = case_when(tmax > 60 & precip == 0 ~ "warm_sunny",
#                                tmax < 40 & (precip == 0 | snow == 0) ~ "cold_sunny", 
#                                tmax > 60 & precip != 0 ~ "warm_cloudy", 
#                                tmax < 40 & (precip != 0 | snow != 0) ~ "cold_cloudy", 
#                                tmax >= 40 & tmax <= 60 ~ "average"), 
#          steps_desc = case_when(steps_daily > 10000 ~ "super_active",
#                                 steps_daily <= 10000 & steps_daily >= 5000 ~ "active",
#                                 steps_daily < 5000 ~ "lazy"))
# # daily_data |> 
# #   mutate(instrumentalness = case_when(instrumentalness_daily >= 0.5 ~ TRUE,
# #                                       TRUE ~ FALSE), 
# #          mins_daily = case_when)
# 
#          
# 
# daily_data <- daily_data |>
#   mutate(weather_desc = relevel(x = factor(daily_data$weather_desc), ref = "average"),
#          steps_desc = relevel(x = factor(daily_data$steps_desc), ref = "lazy"))
# ```
# 
# 
# Top Artist Analysis (Popularity):
# ```{r top artists1}
# top_artists <- spotifyr::get_my_top_artists_or_tracks(type = "artists", 20) |>
#   filter(name != "We Butter The Bread With Butter")
# 
# top_artist_audio_features <- spotify |>
#   filter(master_metadata_album_artist_name %in% top_artists$name) |>
#   group_by(master_metadata_album_artist_name) |>
#   summarise(danceability_mean = mean(danceability), 
#             energy_mean = mean(energy),
#             loudness_norm_mean = mean(loudness_norm), 
#             speechiness_mean = mean(speechiness), 
#             acousticness_mean = mean(acousticness),
#             instrumentalness_mean = mean(instrumentalness),
#             liveness_mean = mean(liveness),
#             valence_mean = mean(valence), 
#             tempo_norm_mean = mean(tempo_norm)) |>
#   bind_cols(popularity = top_artists$popularity)
# 
# 
# top_artist_audio_features |>
#   pivot_longer(2:10) |>
#   ggplot(aes(x = popularity, y = value)) +
#   geom_point() + 
#   facet_wrap(~name)
# ```
# 
# 
# 
# 
# ##MLR 
# 
# ```{r}
# lmod_steps <- lm(steps_daily ~ ., data = train_daily)
# summary(lmod_steps)
# ```
# 
# 
# Test for multicollinearity:
# ```{r}
# car::vif(lmod_steps)
# #Energy and loudness appear to be correlated according to VIF > 10
# #Acousticness has VIF > 5
# kappa(lmod_steps)
# #Kappa << 30, so we do not have evidence of multicollinearity
# 
# corr_matrix <- Hmisc::rcorr(as.matrix(train_daily))
# corr_matrix_p <- corr_matrix[[3]]
# corr_matrix_r <- corr_matrix[[1]]
# ```
# 
# I clearly have a problem with multicollinearity, with my VIF scores for loudness and energy being much too large, and acousticness being slightly too large. Using the correlation matrix, I have identified strong negative correlations between the Energy/Loudness predictors and the instrumentalness/acousticness predictors, and strong positive correlations between each of those pairs. 
# 
# To maybe try to start to remedy this problem, I would propose removing either energy or loudness, since they have the strongest pairwise correlation and seem the most similar. Nothing in the audio feature description provided by Spotify indicates that, for example, loudness is input into the Energy score, but since loudness is simply based on decibels, I might expect the energy score to have more wrapped up into it. Therefore, I will remove loudness from this analysis first, and redo the multicollinearity tests. 
# 
# MLR with the loudness_norm_daily predictor removed:
# ```{r}
# train_daily <- train_daily |>
#   select(-c(loudness_norm_daily))
# 
# lmod_steps <- lm(steps_daily ~ ., train_daily)
# summary(lmod_steps)
# 
# performance::check_model(lmod_steps)
# 
# ```
# 
# The `check_model` function indicates there are still some potential collinearity problems, so I will take another look! 
# 
# Test for multicollinearity without the loudness_norm_daily predictor:
# ```{r}
# car::vif(lmod_steps)
# #Energy and acousticness appear to be correlated according to VIF > 5
# kappa(lmod_steps)
# #Kappa << 30, so we do not have evidence of multicollinearity
# 
# corr_matrix <- Hmisc::rcorr(as.matrix(train_daily))
# corr_matrix_p <- corr_matrix[[3]]
# corr_matrix_r <- corr_matrix[[1]]
# ```
# 
# Again, I see that instrumentalness and acousticness have strong correlations with each other. Given this, I will remove acousticness. Since I am trying to model how steps might be explained by audio features, and energy and acousticness appear to be total opposites, having only one of these predictors should suit my purposes. 
# 
# MLR with the loudness_norm_daily & acousticness_daily predictor removed:
# ```{r}
# train_daily <- train_daily |>
#   select(-c(acousticness_daily))
# 
# lmod_steps <- lm(steps_daily ~ ., train_daily)
# summary(lmod_steps)
# 
# performance::check_model(lmod_steps)
# ```
# 
# Test for multicollinearity without the loudness_norm_daily & acousticness_daily predictors:
# ```{r}
# car::vif(lmod_steps)
# #All VIF < 5, so we do not have evidence of multicollinearity
# kappa(lmod_steps)
# #Kappa << 30, so we do not have evidence of multicollinearity
# 
# corr_matrix <- Hmisc::rcorr(as.matrix(train_daily))
# corr_matrix_r <- corr_matrix[[1]]
# ```
# 
# 
# Is there a correlation between how I ended a track and their audio features?
# ```{r}
# spotify <- spotify |>
#   #mutate(reason_end = as.factor(reason_end)) |>
#   mutate(reason_end = relevel(x = factor(spotify$reason_end), ref = "trackdone"))
# 
# summary(lm(loudness_norm ~ reason_end, data = spotify))
# unique(spotify$reason_end)
# ```
# Not an obvious one. 
# 
# 
# # 2019, the year of the podcast! cool to do that analysis later bb 
# ```{r}
# spotify |>
#   separate(date, into = c("year", "month", "day"), sep = "-") |>
#   group_by(year) |>
#   summarize(total_mins = sum(ms_played)/1000) |>
#   ggplot(aes(x = year, y = total_mins)) + 
#            geom_col()
# ```
# 
# # Principal Component Analysis (hh) 
# ```{r}
# # train_pc <- princomp(na.omit(train_daily[,1:10]), cor = TRUE)
# # train_pc
# # summary(train_pc)
# # 
# # round(cor(na.omit(train_daily[,1:10]), train_pc$scores), 3)
# # 
# # #Scree Plot
# # screeplot(train_pc, type = "l")
# # 
# # #I identify 2 components
# # 
# # plot(train_pc$scores[, 1:2], type = 'n', xlab = "PC1", ylab = "PC2")
# # points(train_pc$scores[,1:2], cex = 0.5)
# 
# test <- t(na.omit(train_daily[,1:9]))
# 
# pca <- prcomp(test, scale = FALSE) #scale = FALSE because I already scaled my data 
# pca
# summary(pca)
# plot(pca, type ="l")
# 
# biplot(pca, scale = 0)
# 
# str(pca)
# train_daily_pc <- bind_cols(drop_na(train_daily, 1:10), pca$x[,1:2])
# 
# ggplot(train_daily_pc, aes(x = PC1, y = PC2)) +
#   stat_ellipse(geom = "polygon", color = "black", alpha = 0.5) +
#   geom_point(shape = 21, color = "black")
# 
# cor(drop_na(train_daily[,1:10], 1:10), train_daily_pc[,12:13])
#   
# ```
# 
# 
# Factor Analysis
# ```{r}
# fa_daily <- na.omit(daily_data_scaled[3:11])
# summary(fa_daily)
# library(nFactors)
# 
# ev <- eigen(cor(fa_daily))
# nS <- nScree(x = ev$values)
# plotnScree(nS, legend = FALSE) #Keep factors greater than eigen values > 1 (ie. above green triangles 
# 
# print(ev$values) #Keep first 3 factors (?)
# 
# fit <- factanal(fa_daily, 3, rotation = "varimax")
# print(fit, digits = 2, cutoff = 0.3, sort = T)
# 
# #Uniqueness: how strongly the item loads on its own own factor and not on other factors
# #
# ```
# 
# 
# Bitch what the FUCK 
# 
# ```{r}
# ggplot(spotify_daily, aes(x = ))
# ggridges::geom_ridgeline()
# ```
# 
# 
# # Ridge Regression failure 
# ```{r}
# lambda <- seq(0.001, 100, 0.01)
# train_daily_na <- drop_na(train_daily)
# test_daily_na <- drop_na(test_daily)[,2:12]
# ridgefit <- glmnet::glmnet(as.matrix(train_daily_na[,1:10]), as.matrix(train_daily_na[,11]), alpha = 0, lambda = lambda)
# summary(ridgefit)
# plot(ridgefit, xvar = 'lambda', label = TRUE)
# 
# plot(ridgefit, xvar = 'dev', label = TRUE)
# 
# y_predicted <- predict(ridgefit, s = min(lambda), newx = as.matrix(test_daily_na[,1:10]))
# y_predicted
# 
# sst <- sum((as.matrix(test_daily_na[,11]) - mean(as.matrix(test_daily_na[,11])))^2)
# sse <- sum((y_predicted - as.matrix(test_daily_na[,11]))^2)
# 
# rsq <- 1 - (sse/sst)
# 
# MSE = (sum((y_predicted - as.matrix(test_daily_na[,11]))^2) / length(y_predicted))
# MSE
# 
# plot(as.matrix(test_daily_na[,11]), y_predicted)
# #should be a 1:1 relationship lmfao 
# 
# ```
# 
# Wanna do an ANOVA so I'm GONNA:
#   ```{r}
# daily_data_test <- daily_data |> 
#   mutate(precip = naniar::impute_median(precip)) |> 
#   mutate(weather_desc = case_when(tmax < 32 & (precip != 0 | snow != 0) ~ "winter_wet", 
#                                   tmax > 75 & (precip == 0 | snow == 0) ~ "summer_sun", 
#                                   TRUE ~ "transitional")) |> 
#   filter(weather_desc != "transitional")
# #mutate(weather_desc = relevel(factor(weather_desc), ref = "transitional"))
# 
# # tmax > 60 & precip == 0 ~ "warm_sunny",
# #                       tmax < 40 & (precip == 0 | snow == 0) ~ "cold_sunny", 
# #                       tmax > 60 & precip != 0 ~ "warm_cloudy", 
# #                       tmax < 40 & (precip != 0 | snow != 0) ~ "cold_cloudy", 
# #                       tmax >= 40 & tmax <= 60 ~ "average")
# 
# test <- scale(daily_data_test[2:18])
# test <- cbind(test, daily_data_test[,19])
# 
# summary(lm(mins_daily ~ weather_desc, data = daily_data_test)) 
# 
# ggplot(daily_data_test, aes(x = weather_desc, y = mins_daily)) +
#   geom_violin()
# 
# #before scaling, the intercept was significant. after scaling, nothing. lol. 
# 
# ```
# 
# 
# 
# 
# # Playing with LOESS 
# ```{r}
# mins_played <- spotify |> 
#   separate(date, into = c("year", "month", "day"), sep = "-") |> 
#   group_by(month, year) |> 
#   summarize(play_time = sum(ms_played)/60000) |> 
#   tidyr::unite("month_year", month, year, sep = "-") |> 
#   mutate(month_year = lubridate::my(month_year))
# 
# 
# ggplot(mins_played, aes(x = month_year, y = play_time)) + 
#   geom_smooth(span = 0.6) +
#   geom_point() 
# 
# 
# spotify |> 
#   separate(date, into = c("year", "month", "day"), sep = "-") |> 
#   group_by(month, year) |> 
#   summarize(energy = mean(energy), 
#             instrumentalness = mean(instrumentalness)) |> 
#   tidyr::unite("month_year", month, year, sep = "-") |> 
#   mutate(month_year = lubridate::my(month_year)) |> 
#   ggplot() +
#   geom_smooth(aes(x = month_year, y = energy), method="gam", formula = y ~s(x)) +
#   geom_smooth(aes(x = month_year, y = instrumentalness), method="gam", formula = y ~s(x)) +
#   geom_point(aes(x = month_year, y = energy)) +
#   geom_point(aes(x = month_year, y = instrumentalness))
# 
# ```
# 
# 
# Top Artist Analysis (Popularity):
#   ```{r top artists}
# top_100 <- spotify |> 
#   group_by(master_metadata_album_artist_name) |> 
#   summarize(mins_played = sum(ms_played)/60000) |> 
#   slice_max(order_by = mins_played, n = 100) 
# 
# top_10 <- spotify |> 
#   group_by(master_metadata_album_artist_name) |> 
#   summarize(mins_played = sum(ms_played)/60000) |> 
#   slice_max(order_by = mins_played, n = 10) 
# 
# test <- spotify |> 
#   filter(master_metadata_album_artist_name %in% top_100$master_metadata_album_artist_name) |> 
#   mutate(top_10 = case_when(master_metadata_album_artist_name %in% top_10$master_metadata_album_artist_name ~ TRUE, 
#                             TRUE ~ FALSE)) |> 
#   select(artist_name = master_metadata_album_artist_name, date, ms_played, top_10) |> 
#   separate(date, into = c("year", "month", "day"), sep = "-") |> 
#   unite("date", month, year) |> 
#   mutate(date = lubridate::my(date)) |> 
#   group_by(date, top_10) |>
#   summarize(mins_played = sum(ms_played)/60000)
# 
# 
# #would like to plot something that conveys the % of listening time per month that is dedicated to my top 10 
# 
# ggplot(test, aes(x = date, y = ms_played, color = top_10)) + 
#   geom_point()
# 
# 
# 
# ```
# 
# 
# 
# 
# 
# b. Question 2. - ANOVA
# a) Creating factors 
# --> should I center the data and then do ± 1sd for weather extremes? 
#   
#   b) fitting model 
# 
# c) checking assumptions/fit/interpretation
# 
# c. Question 3. - A Quick Foray 
# 
# - Why am I doing this? Well, the spotify data set is LORGE and I found a few interesting patterns. Decided to try more advanced/unknown methods on them but didn't get very far. SO. 
# 
# 1. PCA
# 
# 2. GAM
# 
# Constructing a multiple linear regression to explore the question: 
# Can a specific combination of Spotify attributes (eg. audio features, minutes listened) explain the number of steps I might take in a day?
# 
# 
# # Questions 
# 1. Is there a correlation between the weather (temperature) and the amount of music I listen to? 
# NO
# 
# 2. Does the valence (mood) score correlate with precipitation or temperature? 
# temp- no 
# precip- no 
# 
# 3. Does the loudness/energy score correlate with the amount of activity?
# loudness- pretty constant, but outliers to low loudness only occur on low activity days. 
# energy- similar, but i have a few outliers to lower (0.5) energy days on high activity days.
# 
# 4. Does any audio feature correlate with activity better than others? 
# Maaaaaaybe danceability if I squint 
# 
# 5. Do deviations from my average music taste* increase with deviations in sleep, exercise, or temperature?
# *music taste defined by the mean, normalized score of all audio features.
# 6. Does the popularity score correlate with other audio features? 
# 7. Is there a seasonal pattern to the amount of music I listen to? (basic time series analysis)
# 
# 
# For least squares I have to assess: 
# 
# 1. $E(\varepsilon_i) = 0$ for all $i = 1,...n$
# - The expected value of the residuals is 0. 
# 
# 2. $E(Y_i) = \mathbf{x_i^T\beta}$ for all $i = 1,...,n$ 
# - Linearity
# 
# 3. $Cov(\varepsilon_i, \varepsilon_j) = 0$ for $i \neq j$ and $ = \sigma^2$ for $i = j$
# - Error term is independently distributed and not correlated
# - Homoskedasticity
# 
# 4. $(X^TX)^-1$ exists 
# - Non-collinearity
# 
# 5. $\varepsilon_i \stackrel{iid}{\sim} N(0, \sigma^2)$
# - Residuals are normally distributed 


```{r read tidy weather data, include = FALSE}
# weather_raw <- read_csv("data/weather_2016-2022.csv")
# 
# weather <- weather_raw |> 
#   naniar::replace_with_na_all(condition = ~.x == -999.0) |>
#   unite("date", 1:3, sep = "-") |>
#   mutate(date = lubridate::ymd(date))
# 
# # This implies 4 dates are NA, but when I locate them according to their row index, the date is NOT missing...so. I'leep them. Other NA's exist for data that is missing for precip, snow, snowcover but since these are random days on which instruments were not recording, I don't see why they should be removed. 
# apply(weather, 2, VIM::countNA)
# 
# weather <- weather |>
#   #drop_na(date) |>
#   mutate(tmean = (tmax+tmin)/2) |> 
#   filter(date >= "2016-03-01")
# 
# weather <- naniar::impute_median_at(weather, .vars = c("precip", "snow", "snowcover"))
```
# 

# ```{r trying out some linear algebra, include = FALSE}
# A <- pca_tidy_coef |> 
#   pivot_wider(names_from = component, values_from = value) |> 
#   select(-id) |> 
#   remove_rownames() |> 
#   column_to_rownames(var="terms") |> 
#   as.matrix()
# 
# Sx <- cov(train_daily_pc) #var covar matrix 
# 
# Sy <- A %*% Sx %*% t(A) 
# 
# diag(Sy)
```

