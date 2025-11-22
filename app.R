library(shiny)
library(ggplot2)
library(keras3)
library(png)
library(jpeg)

model_file <- "mnist_model.keras"
if (!file.exists(model_file)) stop("Model file 'mnist_model.keras' not found. Run train_mnist.R first.")
model <- load_model(model_file)

preprocess_upload <- function(img_path) {
  if (!requireNamespace("imager", quietly = TRUE)) {
    stop("Please install package 'imager' (install.packages('imager')) to enable upload processing.")
  }
  ext <- tolower(tools::file_ext(img_path))
  if (ext %in% c("png")) {
    img <- png::readPNG(img_path)
  } else if (ext %in% c("jpg","jpeg")) {
    img <- jpeg::readJPEG(img_path)
  } else {
    stop("Unsupported image format. Use PNG or JPG/JPEG.")
  }
  
  if (length(dim(img)) == 3) {
    img <- apply(img, c(1,2), mean)
  }
  
  img_cimg <- imager::as.cimg(img)
  img_resized <- imager::resize(img_cimg, size_x = 28, size_y = 28)
  m <- as.matrix(img_resized)
  
  if (mean(m) > 0.5) m <- 1 - m
  arr <- as.numeric(m)
  array(arr, dim = c(1,28,28,1))
}

ui <- fluidPage(
  tags$h3("MNIST Upload → Predict"),
  fileInput("upload", "Upload a handwritten digit (PNG / JPG)", accept = c(".png",".jpg",".jpeg")),
  uiOutput("uploaded_plot_ui"),
  verbatimTextOutput("upload_pred_text"),
  plotOutput("upload_prob_bar", height = "220px")
)

server <- function(input, output, session) {
  
  uploaded_processed <- reactive({
    f <- input$upload
    if (is.null(f)) return(NULL)
    tmp <- f$datapath
    tryCatch({
      preprocess_upload(tmp)
    }, error = function(e) {
      showNotification(paste0("Processing failed: ", e$message), type = "error")
      NULL
    })
  })
  
  output$uploaded_plot_ui <- renderUI({
    if (is.null(input$upload)) {
      tags$p("No image uploaded yet.")
    } else {
      plotOutput("uploaded_plot", height = "220px")
    }
  })
  
  output$uploaded_plot <- renderPlot({
    arr <- uploaded_processed()
    if (is.null(arr)) return(NULL)
    m <- matrix(as.numeric(arr[1,,,1]), nrow = 28, byrow = TRUE)
    m_rot <- t(apply(m, 2, rev))
    df <- expand.grid(x = 1:28, y = 1:28)
    df$val <- as.vector(m_rot)
    ggplot(df, aes(x = x, y = y, fill = val)) +
      geom_raster() + scale_y_reverse() +
      scale_fill_gradient(low = "white", high = "black") +
      theme_void() + ggtitle("Uploaded (resized to 28×28)") +
      theme(plot.title = element_text(hjust = 0.5))
  })
  
  output$upload_pred_text <- renderText({
    arr <- uploaded_processed()
    if (is.null(arr)) return("Upload an image to get a prediction.")
    preds <- model %>% predict(arr)
    pl <- which.max(preds) - 1L
    sprintf("Predicted digit: %d\nConfidence: %.2f%%", pl, 100 * max(preds))
  })
  
  output$upload_prob_bar <- renderPlot({
    arr <- uploaded_processed()
    if (is.null(arr)) return(NULL)
    preds <- as.numeric((model %>% predict(arr))[1, ])
    df <- data.frame(digit = 0:9, prob = preds)
    ggplot(df, aes(x = factor(digit), y = prob)) +
      geom_col() + ylim(0,1) +
      labs(x = "Digit", y = "Probability") +
      theme_minimal()
  })
}

shinyApp(ui, server)
