#' @export
makeRLearner.reinfLearn.qlearning = function() {
  makeRLearnerReinfLearn(
    cl = "reinfLearn.qlearning",
    package = "reinforcelearn",
    par.set = makeParamSet(
      makeNumericLearnerParam(id = "n.episodes", default = 100, lower = 1, tunable = FALSE),
      makeNumericLearnerParam(id = "discount", default = 1, lower = 0, upper = 1, tunable = FALSE),
      makeNumericLearnerParam(id = "epsilon", default = 0.1, lower = 0, upper = 1, tunable = TRUE),
      makeNumericLearnerParam(id = "learning.rate", default = 0.1, lower = 0, upper = 1, tunable = TRUE),
      makeNumericLearnerParam(id = "lambda", default = 0, lower = 0, upper = 1, tunable = TRUE),
      makeLogicalLearnerParam(id = "double.learning", default = FALSE, tunable = TRUE)
    ),
    properties = c("episodic"),
    name = "Q-Learning",
    short.name = "qlearning",
    callees = c("qlearning")
  )
}

#' @export
trainLearner.reinfLearn.qlearning = function(.learner, .task, ...) {
  reinforcelearn::qlearning(.task$envir, ...)
}

#' @export
predictLearner.reinfLearn.qlearning = function(.learner, .model, .newdata, ...) {
  print("Not implemented")
}
