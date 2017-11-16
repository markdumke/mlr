#' @rdname Task
#' @export
makeReinfLearnTask = function(id, envir) {
  assertString(id)
  assertClass(env, "R6")
  task = makeS3Obj("Task",
    type = "reinfLearn",
    task.desc = NA
  )
  task$task.desc = makeReinfLearnTaskDesc(id, envir)
  addClasses(task, "ReinfLearnTask")
}

makeReinfLearnTaskDesc = function(id, envir) {
  td = makeReinfLearnTaskDescInternal("reinfLearn", id, envir)
  return(addClasses(td, c("reinfLearnTaskDesc")))
}

makeReinfLearnTaskDescInternal = function(type, id, envir) {
  makeS3Obj("TaskDesc",
    id = id,
    type = type,
    envir = envir,
    states = envir$n.states,
    actions = envir$n.actions
  )
}

#' @export
print.ReinfLearnTask = function(x, ...) {
  td = x$task.desc
  catf("Reinforcement learning task: %s", td$id)
  catf("Type: %s", td$type)
  catf("States: %i", td$states)
  catf("Actions: %i", td$actions)
}
