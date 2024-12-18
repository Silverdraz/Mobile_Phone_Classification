variable "aws_region" {
    description = "AWS Region to create resources"
    default = "ap-southeast-2"
}

variable "project_id" {
    description = "project_id"
    default = "mobile-phone"
}   

variable "model_bucket" {
    description = "s3_bucket"
}

variable "lambda_function_local_path" {
  description = ""
}

variable "docker_image_local_path" {
  description = ""
}

variable "ecr_repo_name" {
  description = ""
}

variable "lambda_function_name" {
  description = ""
}