args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args) >= 1) args[[1]] else "field.csv"
output <- if (length(args) >= 2) args[[2]] else "field_validation.png"
d <- read.csv(input, check.names = FALSE)
x <- sort(unique(d$x))
y <- sort(unique(d$y))
nx <- length(x)
ny <- length(y)
if (nx * ny != nrow(d)) stop("CSV is not a complete rectangular grid")

as_grid <- function(name) {
  # CSV rows have x varying fastest; image() expects z[x,y].
  t(matrix(d[[name]], nrow = ny, ncol = nx, byrow = TRUE))
}

draw_panel <- function(name, title, palette) {
  z <- as_grid(name)
  image(x, y, z, col = palette(256), useRaster = FALSE,
        xlab = "x", ylab = "y", main = title, asp = 1)
  contour(x, y, z, add = TRUE, drawlabels = FALSE,
          col = adjustcolor("black", alpha.f = 0.28))
}

rmse <- sqrt(mean(d$u_error^2))
max_error <- max(d$abs_u_error)
png(output, width = 2400, height = 1450, res = 180)
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0))
draw_panel("u", "Monte Carlo potential", hcl.colors)
draw_panel("u_true", "Analytic potential x^2-y^2", hcl.colors)
draw_panel("u_error", "Signed potential error", colorRampPalette(c("navy", "white", "firebrick")))
draw_panel("abs_u_error", "Absolute potential error", function(n) hcl.colors(n, "Inferno"))
draw_panel("Ex", "Computed field Ex", hcl.colors)
draw_panel("Ey", "Computed field Ey", hcl.colors)
mtext(sprintf("Unit-square validation: RMSE=%.4e, max |error|=%.4e",
              rmse, max_error), outer = TRUE, cex = 1.2)
dev.off()
cat(sprintf("Wrote %s\n", output))
