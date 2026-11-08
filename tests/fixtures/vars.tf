variable "prefix" {
  type    = string
  default = "x"
}

variable "enabled" {
  default = true
}

output "bucket_arn" {
  value = aws_s3_bucket.this.arn
}

locals {
  tags = { Name = var.prefix }
}

locals {
  other = "${local.tags.Name}-suffix"
  third = 3
}
