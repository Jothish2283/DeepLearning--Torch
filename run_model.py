from tqdm.auto import tqdm
import time
import torch
from torch.utils.tensorboard import SummaryWriter

def acc_fn(y_p, y):
  correct=torch.eq(y_p, y).sum().item()
  acc=correct/len(y_p)
  return acc

def run_model(model, epochs, train_dataloader, test_dataloader, device, log_path=None):

  loss_fn=torch.nn.CrossEntropyLoss()
  optimizer=torch.optim.Adam(params=model.parameters())
  epochs=epochs


  train_loss_c, train_acc_c, test_loss_c, test_acc_c=[],[],[],[]
  start_t=time.perf_counter()
  for epoch in tqdm(range(epochs)):
    acc, loss=0, 0
    for x, y in train_dataloader:
      x, y=x.to(device), y.to(device)
      model.train()
      y_ps=model(x)
      y_p=y_ps.argmax(dim=1)
      
      Loss=loss_fn(y_ps, y)
      acc+=acc_fn(y_p, y)
      loss+=Loss.item()

      optimizer.zero_grad()

      Loss.backward()

      optimizer.step()

    train_loss, train_acc= loss/len(train_dataloader), acc/len(train_dataloader)
    train_loss_c.append(train_loss)
    train_acc_c.append(train_acc)
    acc, loss=0, 0

    for x, y in test_dataloader:
      model.eval()
      x, y= x.to(device), y.to(device)
      with torch.inference_mode():
            y_ps=model(x)
            y_p=y_ps.argmax(dim=1)
            
            Loss=loss_fn(y_ps, y)
            acc+=acc_fn(y_p, y)
            loss+=Loss.item()
    
    test_loss, test_acc= loss/len(test_dataloader), acc/len(test_dataloader)

    if log_path:
        writer=SummaryWriter(log_path)
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"Train_loss": train_loss, "Test_loss": test_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Acc", tag_scalar_dict={"Train_acc": train_acc, "Test_acc": test_acc}, global_step=epoch)

    test_loss_c.append(test_loss)
    test_acc_c.append(test_acc)
    print(f"\n--Epoch: {epoch}--")
    print(f"Train_loss:{train_loss :.3f} | Train_acc:{train_acc :.3f} | Test_loss:{test_loss :.3f} | Test_acc:{test_acc :.3f}")

  if log_path:
    writer.close()
  stop_t=time.perf_counter()
  results={"train_loss":train_loss_c, "train_acc":train_acc_c, "val_loss":test_loss_c, "val_acc":test_acc_c}
  print(f"\nTime Taken: {(stop_t-start_t)/60} mins")
  return results
