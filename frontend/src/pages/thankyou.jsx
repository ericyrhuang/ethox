
import { useNavigate } from 'react-router-dom';
import '../styles/thankyou.css';

function Thankyou(){
    const navigate = useNavigate();
    return(
        <div className="thankyou-container">
            <h1>Thank you for your Listing</h1>
            <button onClick={() => navigate('/marketplace')}>Go to Marketplace</button>
        </div>
    );
}

export default Thankyou;