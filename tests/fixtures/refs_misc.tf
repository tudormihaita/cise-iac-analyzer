resource "aws_lambda_function" "f" {
  function_name = path.module
  count         = count.index
  role          = self.arn
  x             = each.value
  vpc           = module.net.vpc_id
}
