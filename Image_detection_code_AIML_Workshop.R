packages <- c("keras3","tensorflow","caret","reshape2","shiny")
install.packages(packages)
library(shiny)
library(keras3)
library(tensorflow)
library(caret)
library(reshape2)

#######Setting seeds for reproducibility #######################
tf$compat$v1$enable_eager_execution()
set.seed(42)
tf$random$set_seed(42)

# Loading the dataset #######
cifar10 <- dataset_cifar10()
x_train_full <- cifar10$train$x
y_train_full <- cifar10$train$y
x_test_full <- cifar10$test$x
y_test_full <- cifar10$test$y

# CIFAR-10 original class labels######
class_labels <- c("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# We use only 10,000 instances to keep the computations tractable#######
x_train <- x_train_full[1:18000,,,]
y_train <- y_train_full[1:18000]
x_test <- x_test_full[1:2000,,,]
y_test <- y_test_full[1:2000]
x_train <- x_train / 255
x_test <- x_test / 255

# Defining the deep neural network######
cnn_model <- function(conv_layers, dense_layers, learning_rate) {
  tryCatch({
    model <- keras_model_sequential()
    model %>% 
      layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(32, 32, 3), padding= "same", name = "conv2d_1") %>%
      layer_max_pooling_2d(pool_size = c(2, 2), name = "maxpool_1")
    
    for (i in 2:conv_layers) {
      model %>% 
        layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', padding="same", name = paste0("conv2d_", i)) %>%
        layer_max_pooling_2d(pool_size = c(2, 2), name = paste0("maxpool_", i))
    }
    
    model %>% layer_flatten(name = "flatten")
    
    for (i in 1:dense_layers) {
      model %>% layer_dense(units = 64, activation = 'relu', name = paste0("dense_", i))
    }
    
    model %>% layer_dense(units = 10, activation = 'softmax', name = "output")
    
    model %>% compile(
      loss = 'sparse_categorical_crossentropy',
      optimizer = optimizer_adam(learning_rate = learning_rate),
      metrics = 'accuracy'
    )
    
    return(model)
  }, error = function(e) {
    cat("Error in model creation: ", e$message, "\n")
    NULL
  })
}

##############################################################
############# The Shiny App ##################################
##############################################################

ui <- fluidPage(
  titlePanel("Detecting images using the CIFAR-10 dataset"),
  
  sidebarLayout(
    sidebarPanel(
      sliderInput("conv_layers", "Number of Convolutional Layers", min = 2, max = 5, value = 2),
      sliderInput("dense_layers", "Number of Dense Layers", min = 1, max = 5, value = 1),
      sliderInput("learning_rate", "Learning Rate", min = 0.0001, max = 0.01, value = 0.001),
      sliderInput("epochs", "Number of Epochs", min = 1, max = 100, value = 10),
      sliderInput("instance_index", "Which instance do you want to predict?", min = 0, max = 1999, value = 0),
      actionButton("train", "Train and Predict"),
      hr(),
      h3("R code for the Model"),
      verbatimTextOutput("generated_code")  # Display the generated R code
    ),
    
    mainPanel(
      textOutput("accuracy"),
      textOutput("computation_time"),
      plotOutput("original_image"),
      uiOutput("prediction_probabilities"),
      uiOutput("confusion_matrix_ui"),
      plotOutput("training_error", width="500px")
    )
  )
)

server <- function(input, output, session) {
  
  # Reactive expression to generate the R code based on the inputs
  generated_code <- reactive({
    code <- paste0(
      "cnn_model <- function(conv_layers = ", input$conv_layers, ", dense_layers = ", input$dense_layers, ", learning_rate = ", input$learning_rate, ") {\n",
      "  model <- keras_model_sequential()\n",
      "  model %>% \n",
      "    layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(32, 32, 3), padding= 'same') %>%\n",
      "    layer_max_pooling_2d(pool_size = c(2, 2))\n"
    )
    
    for (i in 2:input$conv_layers) {
      code <- paste0(code,
                     "  model %>% \n",
                     "    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', padding='same') %>%\n",
                     "    layer_max_pooling_2d(pool_size = c(2, 2))\n"
      )
    }
    
    code <- paste0(code,
                   "  model %>% layer_flatten()\n"
    )
    
    for (i in 1:input$dense_layers) {
      code <- paste0(code,
                     "  model %>% layer_dense(units = 64, activation = 'relu')\n"
      )
    }
    
    code <- paste0(code,
                   "  model %>% layer_dense(units = 10, activation = 'softmax')\n",
                   "  model %>% compile(\n",
                   "    loss = 'sparse_categorical_crossentropy',\n",
                   "    optimizer = optimizer_adam(learning_rate = ", input$learning_rate, "),\n",
                   "    metrics = 'accuracy'\n",
                   "  )\n",
                   "  return(model)\n",
                   "}\n",
                   "\n",
                   "history <- cnn_model(", input$conv_layers, ", ", input$dense_layers, ", ", input$learning_rate, ") %>% fit(\n",
                   "  x_train, y_train, \n",
                   "  epochs = ", input$epochs, ", \n",
                   "  verbose = 2\n",
                   ")"
    )
    
    code
  })
  
  # Render the generated R code in the UI
  output$generated_code <- renderText({
    generated_code()
  })
  
  observeEvent(input$train, {
    tryCatch({
      model <- cnn_model(input$conv_layers, input$dense_layers, input$learning_rate)
      if (is.null(model)) {
        showNotification("Model creation failed. Check the console for errors.", type = "error")
        return()
      }
      
      withProgress(message = 'Training model...', value = 0, {
        start_time <- Sys.time()
        history <- model %>% fit(
          x_train, y_train, 
          epochs = input$epochs, 
          verbose = 0, 
          callbacks = list(
            callback_lambda(on_epoch_end = function(epoch, logs) {
              incProgress(1 / input$epochs, detail = paste("Epoch", epoch , "of", input$epochs))
            })
          )
        )
        end_time <- Sys.time()
      })
      
      computation_time <- end_time - start_time
      
      scores <- model %>% evaluate(x_test, y_test, verbose = 0)
      
      output$accuracy <- renderText({
        paste("Test Accuracy: ", round(scores$accuracy * 100, 2), "%")
      })
      
      output$computation_time <- renderText({
        paste("Computation Time: ", round(as.numeric(computation_time, units = "secs"), 2), " seconds")
      })
      
      instance_index <- input$instance_index
      instance_image <- array_reshape(x_test[instance_index + 1,,,], c(1, 32, 32, 3))
      instance_label <- y_test[instance_index + 1]
      
      prediction <- model %>% predict(instance_image)
      predicted_class <- which.max(prediction[1,]) - 1
      
      output$original_image <- renderPlot({
        plot(as.raster(instance_image[1,,,]))
        title(main = paste("Original Image. True Label: ", class_labels[instance_label + 1], ", Predicted Label: ", class_labels[predicted_class + 1]))
      })
      
      output$prediction_probabilities <- renderUI({
        probs <- round(prediction[1,], 2)
        tagList(
          tags$h3("Probability of classes for the chosen instance"),
          paste(sapply(1:10, function(i) paste(class_labels[i], ":", probs[i])), collapse = "; ")
        )
      })
      
      y_pred_probs <- model %>% predict(x_test)
      y_pred <- apply(y_pred_probs, 1, which.max) - 1
      
      y_test_labels <- factor(y_test, levels = 0:9, labels = class_labels)
      y_pred_labels <- factor(y_pred, levels = 0:9, labels = class_labels)
      
      output$confusion_matrix_ui <- renderUI({
        tagList(
          tags$h3("Contingency Table with percentages of predictions"),
          tableOutput("confusion_matrix")
        )
      })
      
      output$confusion_matrix <- renderTable({
        cm <- confusionMatrix(y_pred_labels, y_test_labels, dnn = c("Predicted", "True"))
        cm_table <- as.data.frame(cm$table)
        cm_table$Percent <- cm_table$Freq / colSums(cm$table)[cm_table$True] * 100
        cm_table <- dcast(cm_table, True ~ Predicted, value.var = "Percent", fill = 0)
        cm_table <- cbind(cm_table, "Percent" = rowSums(cm_table[,-1]))
        cm_table
      })
      
      output$training_error <- renderPlot({
        par(mar = c(5, 4, 4, 2) + 0.1)  # Adjust margins
        plot(history$metrics$loss, type = "l", col = "blue", lwd = 2, axes = FALSE, ann = FALSE, frame.plot = TRUE)
        title(main = "Training error vs Epochs (You can use this to tune your learninr rate)", xlab = "Epochs", ylab = "Cross Entropy error")
      })
      
    }, error = function(e) {
      cat("Error during training and prediction: ", e$message, "\n")
      showNotification("An error occurred. Check the console for details.", type = "error")
    })
  })
}

shinyApp(ui = ui, server = server)
