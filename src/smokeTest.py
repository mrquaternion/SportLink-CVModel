from fastapi import APIRouter

router = APIRouter()


@router.get("")
def smoke_test():
    return {"message": "OK"}
