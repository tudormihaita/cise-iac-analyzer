# main resources
resource "aws_s3_bucket" "this" {
  bucket = "${var.prefix}-bucket"
  tags   = local.tags
}

resource "aws_s3_bucket_policy" "p" {
  bucket     = aws_s3_bucket.this.id
  policy     = data.aws_iam_policy_document.doc.json
  depends_on = [aws_s3_bucket.this, module.net]
}

data "aws_iam_policy_document" "doc" {
  statement {
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.this.arn}/*"]
  }
}

resource "aws_instance" "web" {
  count         = var.enabled ? 1 : 0
  ami           = "ami-123"
  instance_type = "t3.micro"
  user_data     = <<-EOT
    #!/bin/bash
    echo hello
  EOT
}

resource "aws_security_group" "sg" {
  for_each = var.rules
  name     = each.key
  dynamic "ingress" {
    for_each = each.value
    content {
      from_port = ingress.value
      to_port   = ingress.value
      protocol  = "tcp"
    }
  }
}

module "net" {
  source = "./modules/net"
  cidr   = var.cidr
}
