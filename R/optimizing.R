# ========================================================================*
# Helper functions ----
# ========================================================================*

# Get the value of the objective at the best set of parameters found.
get_val <- function(.op, .pkg) {
    if (.pkg %in% c("stats", "nloptr")) {
        val <- .op$value
    } else {
        val <- .op$fval
    }
    return(val)
}
# Get convergence code
get_converge_code <- function(.op, .pkg) {
    if (.pkg == "minqa") {
        conv_code <- .op$ierr
    } else if (.pkg == "nloptr") {
        conv_code <- .op$convergence
    } else {
        conv_code <- .op$convergence
    }
    return(conv_code)
}
# Get whether an optimization has converged.
get_converged <- function(.op, .pkg) {
    if (.pkg == "minqa") {
        conv <- .op$ierr == 0L
    } else if (.pkg == "nloptr") {
        conv <- .op$convergence > 0L
    } else {
        conv <- .op$convergence == 0L
    }
    return(conv)
}
# Combine arguments and run optimizer
run_optim <- function(.optim, .pars, .fn, .control, .fn_args) {
    .args <- c(list(par = .pars, fn = .fn, control = .control), .fn_args)
    # Adjust arguments across packages:
    if (packageName(environment(.optim)) == "nloptr") {
        if (!is.null(.args[["control"]][["maxit"]])) {
            .args[["control"]][["maxeval"]] <- .args[["control"]][["maxit"]]
            .args[["control"]][["maxit"]] <- NULL
        }
        if (!is.null(.args[["control"]][["reltol"]])) {
            .args[["control"]][["xtol_rel"]] <- .args[["control"]][["reltol"]]
            .args[["control"]][["reltol"]] <- NULL
        }
        .args[["x0"]] <- .pars
        .args[["par"]] <- NULL
    }
    if (packageName(environment(.optim)) == "minqa") {
        if (!is.null(.args[["control"]][["maxit"]])) {
            .args[["control"]][["maxfun"]] <- .args[["control"]][["maxit"]]
            .args[["control"]][["maxit"]] <- NULL
        }
        # No obvious equivalent in `minqa`, so just remove this:
        .args[["control"]][["reltol"]] <- NULL
    }
    op <- do.call(.optim, .args)
    return(op)
}




# Run box evaluations for `n_bevals` per box.
# If not multithreaded (`multithread = FALSE`), then optionally write to file, then return
# row corresponding to the best evaluation.
# If multithreaded (`multithread = TRUE`), then do the same thing if no file is requested.
# If a file is requested, then this function returns a list with the best row and
# the entire matrix, for writing to a file later.
one_bevals <- function(i,
                       n_bevals,
                       n_pars,
                       par_names,
                       mids,
                       steps,
                       fn,
                       fn_args,
                       na_stop,
                       file_name,
                       multithread = FALSE) {

    # Objective function evaluations for box `i`
    bevals_i <- lapply(1:n_bevals, \(j) {
        .pars <- runif(n_pars, mids - steps * i, mids + steps * i)
        .val <- do.call(fn, c(list(.pars), fn_args))
        return(c(i, .pars, .val))
    }) |>
        do.call(what = rbind)
    colnames(bevals_i) <- c("box", par_names, "val")

    if (any(is.na(bevals_i[,"val"])) && na_stop) {
        warning(sprintf(paste("\nThere were NAs in the %ith box.",
                              "The matrix of parameter values and",
                              "evaluations is being returned."), i))
        return(bevals_i)
    }
    if (all(is.na(bevals_i[,"val"]))) {
        warning(sprintf(paste("\nThe %ith box was all NAs!",
                              "The matrix of parameter values and",
                              "evaluations is being returned."), i))
        return(bevals_i)
    }
    if (!is.na(file_name) && !multithread) {
        write.table(bevals_i, file_name, append = i > 1, quote = FALSE,
                    sep = ifelse(file_ext(file_name) == "txt", "\t", ","),
                    row.names = FALSE, col.names = i == 1)
    }
    best_idx <- which(bevals_i[,"val"] == min(bevals_i[,"val"], na.rm = TRUE))[[1]]
    if (!is.na(file_name) && multithread) {
        return(list(best = bevals_i[best_idx, c(par_names, "val")],
                    all = bevals_i))
    }
    return(bevals_i[best_idx, c(par_names, "val")])
}


# Similar to above but generate ALL the box evaluations. Takes
# care of multithreading and file writing.
all_bevals <- function(n_boxes,
                       n_bevals,
                       n_pars,
                       par_names,
                       mids,
                       steps,
                       fn,
                       fn_args,
                       na_stop,
                       file_name,
                       multithread,
                       obj_env,
                       verbose) {

    if (multithread) {

        if (!requireNamespace("mirai", quietly = TRUE)) {
            stop("Package \"mirai\" must be installed to use multiple threads.")
        }
        mirai::require_daemons()
        args_ <- list(n_bevals = n_bevals, n_pars = n_pars,
                      par_names = par_names, mids = mids, steps = steps,
                      fn = fn, fn_args = fn_args, na_stop = na_stop,
                      file_name = file_name, multithread = TRUE)
        one_bevals_mt <- \(.i) do.call(one_bevals, c(list(i = .i), args_))

        # Create new environment for multithreading:
        if (is.null(obj_env)) {
            env <- new.env()
        } else {
            env <- as.environment(as.list(obj_env, all.names=TRUE))
        }
        env$args_ <- args_
        env$one_bevals_mt <- one_bevals_mt
        env$one_bevals <- one_bevals
        env$file_ext <- file_ext

        if (verbose) {
            box_beval_list <- mirai::mirai_map(1:n_boxes, one_bevals_mt,
                                               env)[.progress]
        } else {
            box_beval_list <- mirai::mirai_map(1:n_boxes, one_bevals_mt, env)[]
        }
        if (any(sapply(box_beval_list, is.character))) {
            idx <- which(sapply(box_beval_list, is.character))[[1]]
            stop("\n`mirai::mirai_map()` produced the following error message:\n",
                 box_beval_list[[idx]])
        }
        if (!is.na(file_name)) {
            # Writing to file can't be done asynchronously:
            for (i in 1:n_boxes) {
                write.table(box_beval_list[[i]][["all"]], file_name,
                            append = i > 1, quote = FALSE,
                            sep = ifelse(file_ext(file_name) == "txt", "\t", ","),
                            row.names = FALSE, col.names = i == 1)
            }
            best_box_bevals <- lapply(box_beval_list, \(x) x[["best"]]) |>
                do.call(what = rbind)
        } else {
            best_box_bevals <- do.call(rbind, box_beval_list)
        }

    } else {

        if (verbose) cli_progress_bar("Box evaluations", total = n_boxes)
        best_box_bevals <- matrix(0.0, n_boxes, n_pars+1)
        for (i in 1:n_boxes) {
            best_box_bevals[i,] <- one_bevals(i, n_bevals, n_pars, par_names,
                                              mids, steps, fn, fn_args, na_stop,
                                              file_name)
            if (verbose) cli_progress_update()
        }
        if (verbose) cli_progress_done()

    }
    colnames(best_box_bevals) <- c(par_names, "val")
    # sort with lowest (best fit) at the top:
    best_box_bevals <- best_box_bevals[order(best_box_bevals[,"val"]),]

    return(best_box_bevals)
}




# Run one optimization step.
# Return the matrix of best output from this step.
# Note: for file_name, use `file_names[[k+1]]` bc first file is for box evaluations
one_optim_step <- function(k,
                           op_input,
                           par_names,
                           n_optims,
                           fn,
                           fn_args,
                           pkg,
                           optimizer,
                           control,
                           n,
                           file_name,
                           multithread,
                           obj_env,
                           verbose) {

    if (verbose) message(sprintf("Starting optimization #%i...\n", k))
    # Last optimizer outputs different object type:
    if (k == n_optims) {
        op_out <- \(op, pars) return(op)
        op_bind_sort <- function(x) {
            vals <- sapply(x, \(y) get_val(y, pkg))
            sort_idx <- order(vals)
            x <- x[sort_idx]
            return(x)
        }
    } else {
        op_out <- function(op, pars) {
            row <- rbind(c(pars, op$par, get_converge_code(op, pkg),
                           get_val(op, pkg)))
            colnames(row) <- c(paste0(par_names, "_start"),
                               paste0(par_names, "_end"),
                               "conv_code", "val")
            return(row)

        }
        op_bind_sort <- function(x) {
            x <- do.call(rbind, x)
            x <- x[order(x[,"val"]),]
            return(x)
        }
    }

    if (multithread) {

        # Create new environment for multithreading:
        if (is.null(obj_env)) {
            env <- new.env()
        } else {
            env <- as.environment(as.list(obj_env, all.names=TRUE))
        }
        env$op_input <- op_input
        env$par_names <- par_names
        env$fn <- fn
        env$fn_args <- fn_args
        env$pkg <- pkg
        env$optimizer <- optimizer
        env$control <- control
        env$run_optim <- run_optim
        env$op_out <- op_out
        env$get_converge_code <- get_converge_code
        env$get_val <- get_val

        one_optim <- function(i) {
            pars <- unname(op_input[i,])
            op <- run_optim(optimizer, pars, fn, control, fn_args)
            return(op_out(op, pars))
        }
        if (verbose) {
            optim_out_list <- mirai::mirai_map(1:n, one_optim, env)[.progress]
        } else {
            optim_out_list <- mirai::mirai_map(1:n, one_optim, env)[]
        }

    } else  {

        one_optim <- function(i, .env) {
            pars <- unname(op_input[i,])
            op <- run_optim(optimizer, pars, fn, control, fn_args)
            if (verbose) cli_progress_update(.envir = .env)
            return(op_out(op, pars))
        }
        if (verbose) cli_progress_bar(sprintf("Optimization #%i", k), total = n)
        optim_out_list <- lapply(1:n, one_optim, .env = environment())
        if (verbose) cli_progress_done()
    }
    optim_out <- op_bind_sort(optim_out_list)

    # Optionally write file if not on the final optimization:
    if (k < n_optims && !is.na(file_name)) {
        write.table(optim_out, file_name,
                    quote = FALSE, row.names = FALSE,
                    sep = ifelse(file_ext(file_name) == "txt", "\t", ","))
    }

    return(optim_out)
}



# ========================================================================*
# Main function ----
# ========================================================================*





#' Winnowing optimization
#'
#' @details
#' All output files are tables.
#' The output for the box evaluations has the box number (column `box`),
#' parameter values (named based on bounds arguments or `par1`, `par2`, ...),
#' and output from objective function at those values (`val`).
#' The output for optimizations include the
#' starting parameter values (with `_start` suffix),
#' ending parameter values (with `_end` suffix),
#' convergence code for optimization (`conv_code`), and
#' output from objective function ending parameter values (`val`).
#'
#'
#'
#' @param fn Objective function to minimize.
#' @param lower_bounds Lower bounds of boxes for each parameter.
#'     If this vector has names, these names are used for column names in
#'     intermediate output tables (argument `file_names` below).
#'     If names are present, they must match those in `upper_bounds`,
#'     but don't need to be in the same order.
#' @param upper_bounds Upper bounds of boxes for each parameter.
#'     See `lower_bounds` for providing names for this vector.
#' @param fn_args List containing other arguments to use for `fn`.
#'     Defaults to `list()`.
#' @param n_bevals Number of evaluations of `fn` per box. Defaults to `100L`.
#' @param n_boxes Number of boxes. Defaults to `1000L`.
#' @param n_outputs A numeric vector indicating the number of best-fitting
#'     optimization outputs from each optimization step to pass to the next one.
#'     Because the first step is the optimizations for the best evaluation per
#'     box, the first item in this vector must be `<= n_boxes`.
#'     Because each optimization step is taking a number of fits from the
#'     previous step, each successive item in this vector must be less than
#'     or equal to the previous.
#'     The length of this vector must be `>= 1` and
#'     must match the lengths of `controls`,
#'     `optimizers`, and, if requested, `file_names`.
#'     Defaults `c(100L, 20L, 3L)`.
#' @param controls A list where each item is a named list.
#'     Each list in the top-most list contains arguments to use for
#'     the `control` argument for the optimization that occurs for that step.
#'     The first step is the optimization for each box.
#'     The length of this vector must be `>= 1` and  must match the lengths
#'     of `n_outputs`, `optimizers`, and, if requested, `file_names`.
#'     The default argument for this is
#'     `list(list(maxit = 100, reltol = 1e-4), list(maxit = 500, reltol = 1e-6), list(maxit = 1000, reltol = 1e-8))`.
#'      This results in three optimization steps that get increasingly
#'      polished.
#' @param optimizers A list of optimizer functions to use for each optimization step.
#'     The length of this vector must be `>= 1` and  must match the lengths
#'     of `n_outputs`, `controls`, and, if requested, `file_names`.
#'     The default argument for this is `c(optim, optim, optim)`.
#'     This results in three optimization steps using `stats::optim`.
#' @param file_names A character vector specifying the file name(s)
#'     where to save intermediate output.
#'     The first output is from box evaluations, and the subsequent ones
#'     are from all the optimizations except for the last one.
#'     To output all the results from the last optimizer step,
#'     change the last item of the `n_outputs` argument.
#'     If provided, this vector must be the same length as
#'     `n_outputs`, `controls`, and `optimizers`.
#'     If you want some output to be written and others not to be, just
#'     set the file names for the step(s) you don't want written to `NA`.
#'     For example, for the default 3 optimizations, if
#'     `file_names = c(NA, "file1.txt", "file2.txt")`, the first
#'     two optimization steps will be written to files, but the box
#'     evaluations will not be written.
#'     File names that end with `.txt` will be tab-delimited,
#'     and those that end with `.csv` will be comma-delimited.
#'     See `Details` above for info on the output from these files.
#'     Note that an error will trigger if attempting to overwrite an existing
#'     file unless the `overwrite` argument is set to `TRUE`.
#'     If `NULL` (the default), no output is written.
#' @param overwrite A single logical for whether to allow overwriting files
#'     for intermediate output (the `file_names` argument to this function).
#'     Defaults to `FALSE`.
#' @param na_stop Single logical for whether to return the matrix of initial
#'     evaluations in each box if it contains `NA`s.
#'     If `FALSE`, the optimizer ignores these values and continues on
#'     (unless they're all `NA`).
#'     Defaults to `FALSE`.
#' @param multithread Number of threads to use via the `mirai` package.
#'     Note that `mirai` is a suggested package, so it's not installed with
#'     this package by default.
#'     If `mirai` is not installed, `multithread = TRUE` generates an error.
#'     See examples below and `?mirai::daemons` for how to use multithreading.
#'     Defaults to `FALSE`.
#' @param obj_env Optional list or environment that contains all objects needed
#'     to run the objective function.
#'     This is only used (and necessary) if `multithread = TRUE`.
#'     See examples for how to use this and when it's necessary.
#'     Defaults to `NULL`.
#' @param verbose Logical for whether to print messages when each step is
#'     started. Defaults to `FALSE`.
#'
#'
#' @importFrom stats optim
#' @importFrom tools file_ext
#' @importFrom cli  cli_progress_bar
#' @importFrom cli  cli_progress_update
#' @importFrom cli  cli_progress_done
#'
#' @returns A list containing `tail(n_outputs, 1)` object(s) of the class
#'     returned by the last optimization step.
#'
#'
#' @examples
#' # "Continuous location planning problem with Manhattan metric"
#' # From https://ds-pl-r-book.netlify.app/optimization-in-r.html
#' fn <- function(loc, a, x, y) sum(a * (abs(x - loc[1]) + abs(y - loc[2]) ) )
#' n <- 100
#' a.vec <- sample(1:100, size = n)    # sample weights for each point/customer
#' x.vec <- rnorm(n)                   # sample x coordinates
#' y.vec <- rnorm(n)                   # sample y coordinates
#'
#' res <- winnowing_optim(fn, lower_bounds = rep(-1, 2), upper_bounds = rep(1, 2),
#'                        fn_args = list(a = a.vec, x = x.vec , y = y.vec))
#'
#' \dontrun{
#' # Example using two threads:
#' # install.packages("mirai")
#' # Set mirai daemons and seed for reproducibility:
#' mirai::daemons(2, seed = 1)
#'
#' # If your objective function is from a package or relies on function(s)
#' # from package(s), you should attach those packages for all child processes
#' # using `mirai::everywhere`. Code for loading this package would be:
#' # everywhere({ library(estimatePMR) })
#'
#' # When the objective function doesn't rely on global variables or user-defined
#' # functions, then you can just run it with `multithread = TRUE`:
#' res <- winnowing_optim(fn, lower_bounds = rep(-1, 2), upper_bounds = rep(1, 2),
#'                        fn_args = list(a = a.vec, x = x.vec, y = y.vec),
#'                        multithread = TRUE)
#' # If we define a new version of the objective function that relies on
#' # a user-defined function, `foo`, then we have to provide that function
#' # in the argument `obj_env`:
#' fn2 <- function(loc, a, x, y) sum(a * (foo(x, loc[1]) + foo(y, loc[2]) ) )
#' foo <- function(xy, loc_i) abs(xy - loc_i)
#' res2 <- winnowing_optim(fn2, lower_bounds = rep(-1, 2), upper_bounds = rep(1, 2),
#'                        fn_args = list(a = a.vec, x = x.vec, y = y.vec),
#'                        obj_env = list(foo = foo),
#'                        multithread = TRUE)
#' }
#'
#' @export
#'
winnowing_optim <- function(fn,
                            lower_bounds,
                            upper_bounds,
                            fn_args = list(),
                            n_bevals = 100L,
                            n_boxes = 1000L,
                            n_outputs = c(100L, 20L, 3L),
                            controls = list(list(maxit = 100, reltol = 1e-4),
                                            list(maxit = 500, reltol = 1e-6),
                                            list(maxit = 1000, reltol = 1e-8)),
                            optimizers = c(optim, optim, optim),
                            file_names = NULL,
                            overwrite = FALSE,
                            na_stop = FALSE,
                            multithread = FALSE,
                            obj_env = NULL,
                            verbose = FALSE) {

    # Type and length checks:
    stopifnot(is.function(fn))
    stopifnot(is.numeric(lower_bounds) && is.numeric(upper_bounds))
    stopifnot(length(lower_bounds) == length(upper_bounds))
    stopifnot(is.list(fn_args))
    stopifnot(length(n_bevals) == 1L  && is.numeric(n_bevals) &&
                  as.integer(n_bevals) == n_bevals)
    stopifnot(length(n_boxes) == 1L  && is.numeric(n_boxes) &&
                  as.integer(n_boxes) == n_boxes)
    stopifnot(is.list(controls))
    stopifnot(length(controls) >= 1)
    stopifnot(all(sapply(controls, is.list)))
    stopifnot(is.list(optimizers))
    stopifnot(length(optimizers) == length(controls))
    stopifnot(all(sapply(optimizers, is.function)))
    stopifnot(is.numeric(n_outputs))
    stopifnot(length(n_outputs) == length(controls))
    stopifnot(min(n_outputs) < n_boxes)
    stopifnot(all(diff(n_outputs) < 0))
    stopifnot(length(overwrite) == 1L && inherits(overwrite, "logical"))
    stopifnot(length(na_stop) == 1L && inherits(na_stop, "logical"))
    stopifnot(length(multithread) == 1L && is.logical(multithread))
    stopifnot(is.null(obj_env) || is.list(obj_env) || is.environment(obj_env))
    if (is.list(obj_env)) obj_env <- as.environment(obj_env)

    # Value checks:
    stopifnot(all(lower_bounds < upper_bounds))
    stopifnot(n_bevals > 0L)
    stopifnot(n_boxes > 0L)
    stopifnot(all(n_outputs > 0L))

    # Number of optimization steps (NOT including box evaluations):
    n_optims <- length(controls)

    # Checking for names in one or more `_bounds` objects:
    if ((!is.null(names(lower_bounds)) && is.null(names(upper_bounds))) ||
        (is.null(names(lower_bounds)) && !is.null(names(upper_bounds)))) {
        stop("If `lower_bounds` is named, `upper_bounds` should be too, ",
             "and vice versa")
    }
    # If they are named, make sure they're the same and in the same order,
    # and use these for `par_names` (used for output) if present
    if (!is.null(names(lower_bounds))) {
        par_names <- names(lower_bounds)
        if (!identical(sort(par_names), sort(names(upper_bounds)))) {
            stop("Names for `lower_bounds` and `upper_bounds` must be the same ",
                 "(order doesn't matter)")
        }
        upper_bounds <- upper_bounds[par_names]
    } else par_names <- paste0("par", 1:length(lower_bounds))

    # Check validity of file names:
    # Note: I add an extra NA at the end of file_names bc `file_names[[k+1]]`
    # is used for the optimization steps below. Adding the extra NA means that
    # the last step will never be written which is what we want, since it's
    # output by the function anyway.
    if (is.null(file_names)) {
        file_names <- rep(NA_character_, n_optims+1)
    } else {
        stopifnot(is.character(file_names) && length(file_names) == length(controls))
        file_names <- c(file_names, NA_character_)
    }
    for (i in 1:length(file_names)) {
        file_i <- file_names[[i]]
        if (!is.na(file_i)) {
            if (length(file_i) != 1 || !is.character(file_i)) {
                stop(sprintf("The item `file_names[[%i]]` ('%s') is not NA or a single string",
                             i, file_i))
            }
            if (! file_ext(file_i) %in% c("txt", "csv")) {
                stop(sprintf("Extension for `file_names[[%i]]` ('%s') is not 'txt' or 'csv'",
                             i, file_i))
            }
            if (!dir.exists(dirname(file_i))) {
                stop(sprintf("Directory for `file_names[[%i]]` ('%s') does not exist",
                             i, file_i))
            }
            if (!overwrite && file.exists(file_i)) {
                stop(paste0("File for `file_names[[", i, "]]` ('", file_i,
                            "') already exists and `overwrite` is `FALSE`"))
            }
        }
    }

    pkgs <- lapply(optimizers, \(o) packageName(environment(o)))
    if (!all(pkgs %in% c("stats", "minqa", "nloptr"))) {
        stop(paste("this function only programmed for stats::optim and",
                   "optimizers from minqa and nloptr packages"))
    }


    mids <- (upper_bounds + lower_bounds) / 2
    steps <- (upper_bounds - mids) / n_boxes
    n_pars <- length(mids)

    # ---------------------------------------------------------*
    # Box evaluations (potentially writing output):
    # ---------------------------------------------------------*
    if (verbose) message("Starting box evaluations...\n")
    # Best set of parameters for each box:
    best_box_bevals <- all_bevals(n_boxes, n_bevals, n_pars, par_names, mids,
                                  steps, fn, fn_args, na_stop, file_names[[1]],
                                  multithread, obj_env, verbose)


    # ---------------------------------------------------------*
    # Optimizations and optionally write files
    # ---------------------------------------------------------*
    # Number of optimizations for each step:
    n_optims_vec <- c(n_boxes, head(n_outputs, -1))
    # Because initial box optimizations take a different input:
    op_input <- best_box_bevals[,par_names]
    # Now do optimizations themselves:
    for (k in 1:n_optims) {
        op_out <- one_optim_step(k, op_input, par_names, n_optims,
                                 fn, fn_args, pkgs[[k]],
                                 optimizers[[k]], controls[[k]],
                                 n_optims_vec[[k]], file_names[[k+1]],
                                 multithread, obj_env, verbose)
        if (k < n_optims) op_input <- op_out[,paste0(par_names, "_end")]
    }

    best_ops <- op_out[1:tail(n_outputs, 1)]
    not_conv <- sapply(best_ops, \(o) !get_converged(o, tail(pkgs, 1)))

    for (i in which(not_conv)) {
        warning("Final optimization ", i, " did not converged.")
    }

    return(best_ops)

}



