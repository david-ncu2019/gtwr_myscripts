# CORRECTED GTWR Prediction Function
# Fixed to match gtwr.R algorithm exactly

gtwr.predict <- function(formula, data, predictdata, obs.tv, reg.tv, st.bw, 
                                  kernel = "bisquare", adaptive = FALSE, p = 2, theta = 0,
                                  longlat = FALSE, lamda = 0.05, t.units = "auto", ksi = 0,
                                  dMat1, dMat2, calculate.var = TRUE) {
  
  timings <- list()
  timings[["start"]] <- Sys.time()
  this.call <- match.call()
  
  # Process calibration data (same as original)
  p4s <- as.character(NA)
  if (inherits(data, "Spatial")) {
    p4s <- proj4string(data)
    fd.locat <- coordinates(data)
    data <- as(data, "data.frame")
  } else if (inherits(data, "sf")) {
    fd.locat <- st_coordinates(st_geometry(data))
    data <- st_drop_geometry(data)
  } else {
    stop("Data must be Spatial*DataFrame or sf object")
  }
  
  # Extract model variables (same as original)
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")
  
  y <- model.response(mf, "numeric")
  x <- model.matrix(mt, mf)
  
  fd.n <- nrow(x)
  var.n <- ncol(x)
  inde_vars <- colnames(x)[-1]
  
  # Process prediction data (same as original)
  pd.given <- !missing(predictdata)
  if (pd.given) {
    if (inherits(predictdata, "Spatial")) {
      pd.locat <- coordinates(predictdata)
      predictdata <- as(predictdata, "data.frame")
    } else if (inherits(predictdata, "sf")) {
      pd.locat <- st_coordinates(st_geometry(predictdata))
      predictdata <- st_drop_geometry(predictdata)
    }
    
    x.p <- predictdata[, inde_vars, drop = FALSE]
    x.p <- cbind(rep(1, nrow(x.p)), x.p)
    x.p <- as.matrix(x.p)
    colnames(x.p) <- colnames(x)
    
    if (missing(reg.tv)) {
      stop("reg.tv required when predictdata is provided")
    }
  } else {
    pd.locat <- fd.locat
    x.p <- x
    reg.tv <- obs.tv
    predictdata <- data
  }
  
  pd.n <- nrow(x.p)
  
  # FIXED: Use same distance calculation as gtwr.R if matrices not provided
  if (missing(dMat1)) {
    cat("Calculating spatiotemporal distance matrix...\n")
    if (pd.given) {
      dMat1 <- st.dist(fd.locat, pd.locat, obs.tv, reg.tv, p=p, theta=theta, 
                       longlat=longlat, lamda=lamda, t.units=t.units, ksi=ksi)
    } else {
      dMat1 <- st.dist(fd.locat, obs.tv=obs.tv, p=p, theta=theta, 
                       longlat=longlat, lamda=lamda, t.units=t.units, ksi=ksi)
    }
  }
  
  if (!is.matrix(dMat1) || nrow(dMat1) != fd.n || ncol(dMat1) != pd.n) {
    stop("ERROR: dMat1 dimensions incorrect. Expected ", fd.n, "×", pd.n, " matrix")
  }
  
  # FIXED: Estimate coefficients using EXACT same algorithm as gtwr.R
  betas1 <- matrix(nrow = pd.n, ncol = var.n)
  colnames(betas1) <- colnames(x)
  xtxinv <- array(0, dim = c(pd.n, var.n, var.n))
  
  # Use hatmatrix=FALSE for prediction (matches gtwr.R when regression.points given)
  hatmatrix <- FALSE
  
  for (i in 1:pd.n) {
    st.disti <- dMat1[, i]
    W.i <- gw.weight(st.disti, st.bw, kernel, adaptive)
    
    # FIXED: Use gw_reg with same parameters as gtwr.R
    gw.resi <- gw_reg(x, y, W.i, hatmatrix, i)
    betas1[i, ] <- gw.resi[[1]]
    
    # Store inverse for variance calculation
    wspan <- matrix(1, 1, var.n)
    xtw <- t(x * (W.i %*% wspan))
    xtwx <- xtw %*% x
    xtxinv[i, , ] <- solve(xtwx)
  }
  
  # Calculate predictions
  gw.predict <- gw_fitted(x.p, betas1)
  
  # Calculate prediction variances if requested
  predict.var <- NULL
  sigma.hat <- NA
  
  if (calculate.var) {
    cat("Calculating prediction variances...\n")
    
    # Calculate variance using calibration points (if dMat2 available)
    if (missing(dMat2)) {
      cat("Computing dMat2 for variance calculation...\n")
      dMat2 <- st.dist(fd.locat, obs.tv=obs.tv, p=p, theta=theta, 
                       longlat=longlat, lamda=lamda, t.units=t.units, ksi=ksi)
    }
    
    # Estimate error variance using calibration points
    S <- matrix(nrow = fd.n, ncol = fd.n)
    betas_calib <- matrix(nrow = fd.n, ncol = var.n)
    
    for (j in 1:fd.n) {
      st.distj <- dMat2[, j]
      W.j <- gw.weight(st.distj, st.bw, kernel, adaptive)
      gw.resi.calib <- gw_reg(x, y, W.j, TRUE, j)  # hatmatrix=TRUE for variance
      
      betas_calib[j, ] <- gw.resi.calib[[1]]
      S[j, ] <- gw.resi.calib[[2]]
    }
    
    # Calculate residuals and variance (same as gtwr.R)
    yhat_calib <- gw_fitted(x, betas_calib)
    residuals <- y - yhat_calib
    
    tr_S <- sum(diag(S))
    tr_StS <- sum(S^2)
    edf <- fd.n - 2 * tr_S + tr_StS
    
    RSS_gw <- sum(residuals^2)
    sigma.hat <- RSS_gw / edf
    
    # Calculate prediction variances
    predict.var <- numeric(pd.n)
    
    for (i in 1:pd.n) {
      st.disti <- dMat1[, i]
      W.i <- gw.weight(st.disti, st.bw, kernel, adaptive)
      
      w2 <- W.i * W.i
      w2x <- x * w2
      xtw2x <- t(x) %*% w2x
      
      xtxinv_i <- xtxinv[i, , ]
      s0 <- xtxinv_i %*% xtw2x %*% xtxinv_i
      
      x.pi <- x.p[i, ]
      s1_value <- as.numeric(t(x.pi) %*% s0 %*% x.pi)
      
      predict.var[i] <- sigma.hat * (1 + s1_value)
    }
  }
  
  # Package results (same as original)
  gwr.pred.df <- data.frame(betas1, gw.predict)
  colnames(gwr.pred.df) <- c(paste(colnames(x), "coef", sep = "_"), "gtwr_prediction")
  
  if (calculate.var) {
    gwr.pred.df$predict.var <- predict.var
    gwr.pred.df$predict.se <- sqrt(predict.var)
  }
  
  # Add coordinates and time
  gwr.pred.df <- cbind(gwr.pred.df, pd.locat)
  gwr.pred.df$pred_time <- reg.tv
  
  # Add prediction variables
  for (var in inde_vars) {
    if (var %in% names(predictdata)) {
      gwr.pred.df[[paste0("pred_", var)]] <- predictdata[[var]]
    }
  }
  
  # Add metadata
  gwr.pred.df$bandwidth <- st.bw
  gwr.pred.df$kernel <- kernel
  gwr.pred.df$prediction_time <- Sys.time()
  
  # Create spatial object
  if (!is.na(p4s) && p4s != "NA") {
    SDF <- SpatialPointsDataFrame(coords = pd.locat, data = gwr.pred.df,
                                  proj4string = CRS(p4s), match.ID = FALSE)
  } else {
    SDF <- SpatialPointsDataFrame(coords = pd.locat, data = gwr.pred.df, 
                                  match.ID = FALSE)
  }
  
  timings[["stop"]] <- Sys.time()
  
  GTW.arguments <- list(
    formula = formula,
    st.bw = st.bw,
    kernel = kernel,
    adaptive = adaptive,
    fd.n = fd.n,
    pd.n = pd.n
  )
  
  res <- list(
    GTW.arguments = GTW.arguments,
    SDF = SDF,
    sigma.hat = sigma.hat,
    timings = timings,
    this.call = this.call
  )
  
  class(res) <- "gtwrm.pred"
  return(res)
}