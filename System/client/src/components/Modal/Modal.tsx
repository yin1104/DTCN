import React from 'react';
import { Button, Flex, Dialog } from '@radix-ui/themes';

type ModalProps = {
  visibility: boolean,
  title: string,
  info: string,
  handleVis(): void
};

const Modal: React.FC<ModalProps> = (props) => {
  const { visibility, title, info, handleVis } = props

  return (
    <Dialog.Root open={visibility}>
      <Dialog.Content style={{ maxWidth: 450 }}>
        <Dialog.Title>{title}</Dialog.Title>
        <Dialog.Description size="2" mb="4">
          {info}
        </Dialog.Description>
        <Flex gap="3" mt="4" justify="end">
          <Button onClick={handleVis}>知道了</Button>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  )
}
export default Modal;